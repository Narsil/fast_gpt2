FROM rust:1.66 as build

WORKDIR /usr/src/fast_gpt2
COPY docker/dummy.rs .
COPY Cargo.toml .
COPY Cargo.lock .
COPY build.rs .
RUN sed -i 's#src/main.rs#dummy.rs#' Cargo.toml
RUN cargo build --release
RUN sed -i 's#dummy.rs#src/main.rs#' Cargo.toml
COPY . .

RUN cargo build --release

FROM gcr.io/distroless/cc-debian10

COPY --from=build /usr/src/fast_gpt2/target/release/fast_gpt2 /usr/local/bin/fast_gpt2
COPY tokenizer.json .
COPY model.safetensors .
CMD ["fast_gpt2"]


