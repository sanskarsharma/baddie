FROM postgres:16.3-alpine AS pgvector-builder

RUN apk add git
RUN apk add build-base
RUN apk add clang15
RUN apk add llvm15-dev llvm15

WORKDIR /home
RUN git clone --branch v0.7.2 https://github.com/pgvector/pgvector.git
WORKDIR /home/pgvector
RUN make
RUN make install

FROM postgres:16.3-alpine
COPY --from=pgvector-builder /usr/local/lib/postgresql/bitcode/vector.index.bc /usr/local/lib/postgresql/bitcode/vector.index.bc
COPY --from=pgvector-builder /usr/local/lib/postgresql/vector.so /usr/local/lib/postgresql/vector.so
COPY --from=pgvector-builder /usr/local/share/postgresql/extension /usr/local/share/postgresql/extension
