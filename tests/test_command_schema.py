def test_command_schema_fields():
    from schema.command_models import StructuredCommand

    cmd = StructuredCommand(
        request_id="123",
        action_type="LOG",
        payload={"a": 1},
        human_readable_summary="hi",
    )
    assert cmd.request_id == "123"
    assert cmd.action_type == "LOG"
    assert cmd.payload == {"a": 1}
