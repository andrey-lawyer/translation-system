## Translation System (Java + Go)

A simple **translation microservice system** using **gRPC**, with two main components:

* **Java Backend** – handles client requests and delegates translations to Go service.
* **Go Translator** – gRPC service performing translations (currently mock).

---

## Project Structure

```
translation-system/
├── java-backend/     # Spring Boot backend with gRPC client
├── go-translator/    # Go gRPC translation service
└── proto/            # Protobuf definitions shared by both services
```

---

## How It Works

1. Client sends a translation request (text + target languages) to the **Java backend**.
2. Java backend forwards request to **Go translator** via gRPC.
3. Go translator returns translated texts.
4. Java backend responds to the client with combined translations.

---

## Prerequisites

* **Java 17+** and **Gradle** (for Java backend)
* **Go 1.18+** (for Go translator)
* `protoc` compiler with **Go** and **Java** plugins
* `grpcurl` (optional, for testing gRPC services)

---

## Setup & Run

### 1. Generate Protobuf Code

#### Go (go-translator)

```bash
# Install protoc plugins (once)
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Generate Go code
protoc --go_out=paths=source_relative:go-translator/translatorpb \
       --go-grpc_out=paths=source_relative:go-translator/translatorpb \
       proto/translation.proto
```

#### Java (java-backend)

```bash
# From project root
./java-backend/gradlew -p java-backend generateProto

# Or
cd java-backend
./gradlew generateProto
```

---

### 2. Run Services Locally

#### Go Translator

```bash
cd go-translator
go run main.go
```

#### Java Backend (in a new terminal)

```bash
cd java-backend
./gradlew bootRun
```

---

## Docker: Run Full Project

Requirements: Docker Desktop or compatible.

### 1. Build and start all services

```bash
docker compose up --build
```

### 2. Check services

* React UI: [http://localhost:3000](http://localhost:3000)
* Java REST API: [http://localhost:8080/api/translate](http://localhost:8080/api/translate)
* Go gRPC service (inside Docker network): `go-translator:6565`

### 3. Stop services

```bash
docker compose down
```

### 4. Useful Docker commands

```bash
docker compose build java-backend
docker compose logs -f java-backend
```

> Notes:
>
> * Java protobuf code is generated during `java-backend` Docker build.
> * Go protobuf code already exists in `go-translator/translatorpb`.

---

## Testing

### gRPC (Go Translator)

```bash
grpcurl -plaintext \
  -d '{"text":"hello world","fromLang":"en","langs":["ru","uk"]}' \
  localhost:6565 translator.TranslatorService/Translate
```

Expected response:

```json
{
  "translations": {
    "en": "hello world",
    "ru": "привет мир",
    "uk": "привіт світ"
  }
}
```

### REST (Java Backend)

```bash
curl -X POST http://localhost:8080/api/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "hello world",
    "fromLang": "en",
    "langs": ["ru", "uk"]
  }'
```

---

## Environment Variables (Docker Compose)

| Variable             | Default                 | Description                 |
| -------------------- | ----------------------- | --------------------------- |
| `GO_TRANSLATOR_HOST` | `go-translator`         | Hostname for Java → Go gRPC |
| `GO_TRANSLATOR_PORT` | `6565`                  | Port for Go gRPC service    |
| `FRONTEND_ORIGIN`    | `http://localhost:3000` | For CORS                    |
| `REACT_APP_API_URL`  | `/api/translate`        | Frontend build arg          |
