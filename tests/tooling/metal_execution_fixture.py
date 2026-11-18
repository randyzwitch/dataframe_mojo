"""Small observed-pipeline records for parser and admission contracts."""


def pipeline(**changes):
    return dict(
        dict(
            format="dataframe-pipeline-observation-v1",
            node=1,
            pipeline=1,
            plan_id=0,
            query_id=123,
            execution_attempt=0,
            executions=1,
            result_used=True,
            operator="AGGREGATE",
            result_device="metal",
            device_status="completed",
            fallback_reason="",
            gpu_launches=2,
            gpu_completed_launches=2,
            upload_bytes=32,
            download_bytes=8,
            device_waits=1,
            phase_timing_kind="host_wall",
            context_wall_ms=0.1,
            input_wall_ms=0.2,
            device_work_wall_ms=0.3,
            finish_wall_ms=0.1,
            synchronize_wall_ms=0.2,
        ),
        **changes
    )


def evidence(audit, count=2):
    return audit.summarize(
        [
            dict(status="gpu_observed", pipelines=[pipeline(query_id=index)])
            for index in range(count)
        ],
        count,
    )
