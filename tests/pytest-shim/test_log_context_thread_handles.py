import gepa.optimize_anything as oa


def test_stale_thread_handle_for_reused_ident_is_ignored():
    tid = 12345
    oa._thread_handles[tid] = (-1, "stale-handle")

    assert oa._get_thread_handle(tid) is None
    assert tid not in oa._thread_handles
