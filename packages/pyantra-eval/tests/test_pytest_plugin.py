pytest_plugins = ["pytester"]


def test_fixture_passes_when_expectations_hold(pytester) -> None:
    pytester.makepyfile(
        """
        from pyantra import Run, RunStatus
        from pyantra_eval import expect_completed

        def test_ok(pyantra_evals):
            run = Run(run_id="r1", status=RunStatus.COMPLETED)
            report = pyantra_evals.expect(run, expect_completed())
            assert report.passed
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


def test_fixture_fails_when_expectation_fails(pytester) -> None:
    pytester.makepyfile(
        """
        from pyantra import Run, RunStatus
        from pyantra_eval import expect_completed, expect_visited

        def test_bad(pyantra_evals):
            run = Run(run_id="r1", status=RunStatus.FAILED)
            pyantra_evals.expect(run, expect_completed(), expect_visited("a"))
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1, errors=1)
    assert "pyantra-eval expectations failed" in "\n".join(result.outlines)
    assert "status" in "\n".join(result.outlines)
    assert "visited" in "\n".join(result.outlines)


def test_fixture_aggregates_multiple_expect_calls(pytester) -> None:
    pytester.makepyfile(
        """
        from pyantra import Run, RunStatus
        from pyantra_eval import expect_completed

        def test_ok(pyantra_evals):
            run = Run(run_id="r1", status=RunStatus.COMPLETED)
            pyantra_evals.expect(run, expect_completed())
            pyantra_evals.expect(run, expect_completed())
            assert len(pyantra_evals.results) == 2
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


def test_fixture_passess_with_no_expectations(pytester) -> None:
    pytester.makepyfile(
        """
        def test_ok(pyantra_evals):
            pass
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
