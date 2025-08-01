import random

from openinference.instrumentation import using_session
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from phoenix.otel import register

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
def track_message(role, content):
    with using_session(str(random.randint(0, 100))):
        span = trace.get_current_span()
        span.add_event("Event message: yo yo")
        span.set_attribute("role", role)
        span.set_attribute("content", content)
        print(f"{role.capitalize()}: {content}")


# Example conversation
track_message("user", "What's the weather like in London?")
track_message("assistant", "It's sunny and 25°C in London today.")
track_message("user", "Great, should I wear sunglasses?")
track_message("assistant", "Yes, sunglasses would be a good idea!")
