# Docker Official Images copy of Alpine 3.21, pinned to the verified manifest.
ARG BASE=public.ecr.aws/docker/library/alpine@sha256:bc21487101ae47172509130ee1092cc273a831968d64b617776d1f69c81e42ab
FROM ${BASE}
ARG APK_MIRROR=https://dl-cdn.alpinelinux.org/alpine
RUN apk --repositories-file /dev/null --repository "$APK_MIRROR/v3.21/main" \
        --repository "$APK_MIRROR/v3.21/community" add --no-cache bash python3 jq coreutils diffutils findutils gawk grep sed \
    && addgroup -g 10001 terminal \
    && adduser -D -H -u 10001 -G terminal terminal \
    && mkdir -p /opt/terminal-bin /workspace /corpus \
    && for cmd in python3 cat head tail grep wc sort uniq cut sed awk tr ls find jq diff comm paste join nl tee xargs basename dirname date seq mkdir; do \
        tool="$(command -v "$cmd")" && test -n "$tool" && ln -s "$tool" "/opt/terminal-bin/$cmd" || exit 1; \
    done \
    && ln -s /usr/bin/python3 /opt/terminal-bin/python
COPY entrypoint.py /opt/terminal-entrypoint.py
LABEL org.cardinal.terminal.policy="2"
USER 10001:10001
WORKDIR /workspace
ENTRYPOINT ["/usr/bin/python3", "-I", "/opt/terminal-entrypoint.py"]
