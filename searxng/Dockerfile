FROM searxng/searxng:latest

# Copy custom configuration
COPY searxng_settings.yml /etc/searxng/settings.yml

# Set proper permissions
RUN chown -R searxng:searxng /etc/searxng/

# Expose port
EXPOSE 8080

# Use the default entrypoint from the base image