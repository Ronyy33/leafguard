# LeafGuard web UI: static files served by nginx. Talks to the backend ONLY via REST;
# the API base URL is injected at container start (see docker-entrypoint.d/).
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html pipeline.html manual.html /usr/share/nginx/html/
COPY css/ /usr/share/nginx/html/css/
COPY js/ /usr/share/nginx/html/js/
COPY docker-entrypoint.d/40-leafguard-config.sh /docker-entrypoint.d/
RUN chmod +x /docker-entrypoint.d/40-leafguard-config.sh
EXPOSE 80
HEALTHCHECK --interval=15s --timeout=3s CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
