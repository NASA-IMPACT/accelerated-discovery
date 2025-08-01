import uuid

from openai import OpenAI
from openinference.instrumentation import using_session
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from phoenix.otel import register

session_id = str(uuid.uuid4())


client = OpenAI(
    api_key="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",
)

# configure the Phoenix tracer
tracer_provider = register(
    project_name="akd",  # Default is 'default'
    auto_instrument=True,  # See 'Trace all calls made to a library' below
    endpoint="http://localhost:6006/v1/traces",
)
# tracer = tracer_provider.get_tracer(__name__)
tracer = tracer_provider.get_tracer("agent-session")


@tracer.start_as_current_span(
    name="agent",
    attributes={SpanAttributes.OPENINFERENCE_SPAN_KIND: "agent"},
)
def assistant(
    messages: list[dict],
    session_id: str,
):
    current_span = trace.get_current_span()
    current_span.set_attribute(SpanAttributes.SESSION_ID, session_id)
    current_span.set_attribute(SpanAttributes.INPUT_VALUE, messages[-1].get("content"))

    current_span.set_attribute(
        SpanAttributes.LLM_FUNCTION_CALL,
        "make_it_uppercase " + """{"text": messages[-1].get("content")}""",
    )

    with using_session(session_id):
        response = (
            client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a helpful assistant."}]
                + messages,
            )
            .choices[0]
            .message
        )

    current_span.set_attribute(SpanAttributes.OUTPUT_VALUE, response.content)
    return response


messages = [{"role": "user", "content": "hi! im bob"}]
response = assistant(
    messages,
    session_id=session_id,
)
messages = messages + [response, {"role": "user", "content": "what's my name?"}]
response = assistant(
    messages,
    session_id=session_id,
)
messages = messages + [response, {"role": "user", "content": "what's 4+5?"}]
response = assistant(
    messages,
    session_id=session_id,
)
