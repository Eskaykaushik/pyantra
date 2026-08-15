"""DBOS-backed checkpoint store.

A :class:`~pyantra.checkpoint.base.CheckpointStore` built on DBOS Transact
datasources, so checkpoint writes ride DBOS's durability layer instead of
reinventing it. The store keeps a ``pyantra_checkpoints`` table inside a DBOS
application database and runs every operation as a datasource transaction
(:meth:`dbos.SQLAlchemyDatasource.run_tx_step`), which composes with DBOS
workflows: a pyantra graph running inside a ``@DBOS.workflow`` records its
checkpoints with DBOS's exactly-once guarantees.

Requires ``pip install 'pyantra[dbos]'`` (pulls in ``dbos`` and
``sqlalchemy``). Both are imported lazily, so pyantra core stays
dependency-free unless DBOS is actually used.

Example::

    from dbos import SQLAlchemyDatasource
    from pyantra import DBOSCheckpointStore

    datasource = SQLAlchemyDatasource.create("sqlite:///app.db")
    store = DBOSCheckpointStore(datasource=datasource)

    run = graph.compile().run(state, checkpointer=store, run_id="orders-42")
"""

from __future__ import annotations

import time
from typing import Any

from pyantra.checkpoint.base import Checkpoint, CheckpointStore
from pyantra.checkpoint.serializer import JsonSerializer, Serializer
from pyantra.runtime.errors import PyantraError
from pyantra.state.state import StateT

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {table} (
    run_id     TEXT PRIMARY KEY,
    resume_at  TEXT,
    body       TEXT NOT NULL,
    updated_at REAL NOT NULL
)
"""

_UPSERT = """
INSERT INTO {table} (run_id, resume_at, body, updated_at)
VALUES (:run_id, :resume_at, :body, :updated_at)
ON CONFLICT(run_id) DO UPDATE SET
    resume_at = excluded.resume_at,
    body      = excluded.body,
    updated_at = excluded.updated_at
"""

_SELECT_BODY = "SELECT body FROM {table} WHERE run_id = :run_id"

_DELETE = "DELETE FROM {table} WHERE run_id = :run_id"


def _import_sqlalchemy() -> Any:
    try:
        import sqlalchemy as sa
    except ImportError as exc:
        raise PyantraError(
            "DBOSCheckpointStore requires SQLAlchemy. Install it with "
            "pip install 'pyantra[dbos]'."
        ) from exc
    return sa


def _create_datasource(url: str) -> Any:
    try:
        from dbos import SQLAlchemyDatasource
    except ImportError as exc:
        raise PyantraError(
            "DBOSCheckpointStore requires the 'dbos' package. Install it with "
            "pip install 'pyantra[dbos]'."
        ) from exc
    return SQLAlchemyDatasource.create(url)


class DBOSCheckpointStore(CheckpointStore[StateT]):
    """A checkpoint store persisted through a DBOS datasource.

    ``datasource`` is a :class:`dbos.SQLAlchemyDatasource` created with
    ``SQLAlchemyDatasource.create(url)``; pass ``url`` instead and the
    datasource is created for you. ``serializer`` is pluggable (JSON by
    default), and ``table`` names the checkpoint table.

    Because operations run inside ``run_tx_step``, checkpoint writes made from
    inside a DBOS workflow are recorded with the same exactly-once semantics
    as any other datasource transaction.
    """

    def __init__(
        self,
        *,
        datasource: Any | None = None,
        url: str | None = None,
        serializer: Serializer[StateT] | None = None,
        table: str = "pyantra_checkpoints",
    ) -> None:
        if (datasource is None) == (url is None):
            raise ValueError(
                "DBOSCheckpointStore requires exactly one of datasource or url."
            )
        self._sa = _import_sqlalchemy()
        if datasource is None:
            assert url is not None
            datasource = _create_datasource(url)
        self._datasource = datasource
        self._serializer = serializer if serializer is not None else JsonSerializer()
        self._table = table
        self._ensure_schema()

    def save(self, checkpoint: Checkpoint[StateT]) -> None:
        self._datasource.run_tx_step(None, self._insert, checkpoint)

    def load(self, run_id: str) -> Checkpoint[StateT] | None:
        body = self._datasource.run_tx_step(None, self._select_body, run_id)
        if body is None:
            return None
        return self._serializer.loads(body.encode("latin-1"))

    def delete(self, run_id: str) -> None:
        self._datasource.run_tx_step(None, self._delete, run_id)

    def _ensure_schema(self) -> None:
        self._datasource.run_tx_step(None, self._create_table)

    def _create_table(self) -> None:
        session = self._datasource.sql_session()
        session.execute(
            self._sa.text(_CREATE_TABLE.format(table=self._table))
        )

    def _insert(self, checkpoint: Checkpoint[StateT]) -> None:
        session = self._datasource.sql_session()
        session.execute(
            self._sa.text(_UPSERT.format(table=self._table)),
            {
                "run_id": checkpoint.run_id,
                "resume_at": checkpoint.resume_at,
                "body": self._serializer.dumps(checkpoint).decode("latin-1"),
                "updated_at": time.time(),
            },
        )

    def _select_body(self, run_id: str) -> str | None:
        session = self._datasource.sql_session()
        row = session.execute(
            self._sa.text(_SELECT_BODY.format(table=self._table)),
            {"run_id": run_id},
        ).first()
        return row[0] if row else None

    def _delete(self, run_id: str) -> None:
        session = self._datasource.sql_session()
        session.execute(
            self._sa.text(_DELETE.format(table=self._table)),
            {"run_id": run_id},
        )


__all__ = ["DBOSCheckpointStore"]
