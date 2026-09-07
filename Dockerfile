# GoVecDB as a service.
#
#   docker build -t govecdb .
#   docker run -p 8080:8080 -v govecdb-data:/data \
#     -e GOVECDB_AUTH_TOKEN=... govecdb
#
# The final image is FROM scratch and holds one static binary. That is not
# minimalism for its own sake: the module has no third-party dependencies and no
# CGO, so there is genuinely nothing else to ship, and an image with no shell and
# no package manager has no package manager CVEs to triage every month.

FROM golang:1.25-alpine AS build

# Baked into the binary and reported by /metrics and -version. Passed by CI from
# the tag; "devel" is the honest answer for a local build.
ARG VERSION=devel

WORKDIR /src
COPY . .

# CGO_ENABLED=0 for a genuinely static binary — scratch has no libc to link
# against. -trimpath keeps build paths out of the binary, so the image does not
# describe the machine that produced it.
RUN CGO_ENABLED=0 go build \
      -trimpath \
      -ldflags "-s -w -X main.version=${VERSION}" \
      -o /out/govecdbd ./cmd/govecdbd

# Created here rather than at runtime: the container runs as a non-root user
# that cannot create a directory at the root of a filesystem it does not own.
RUN mkdir -p /data


FROM scratch

COPY --from=build /out/govecdbd /govecdbd
COPY --from=build --chown=65532:65532 /data /data

# Non-root, numeric because scratch has no /etc/passwd to resolve a name in.
USER 65532:65532

# The database directory. Without a volume it lives in the container's writable
# layer and dies with the container, which is rarely what anyone means.
VOLUME ["/data"]

EXPOSE 8080

# 0.0.0.0 rather than the binary's own loopback default: a container that binds
# 127.0.0.1 is reachable from nothing at all. The daemon warns when it serves a
# routable address without GOVECDB_AUTH_TOKEN set, and that warning is worth
# reading rather than silencing.
ENTRYPOINT ["/govecdbd"]
CMD ["-dir", "/data", "-addr", "0.0.0.0:8080"]

# No HEALTHCHECK: scratch has no shell and no curl to run one with. Point the
# orchestrator's own HTTP probe at /healthz for liveness and /readyz for
# readiness — /readyz starts failing first during a graceful shutdown, which is
# the whole reason there are two.
