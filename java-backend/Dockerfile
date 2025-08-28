FROM gradle:8.9.0-jdk21-alpine AS builder

# Устанавливаем bash и совместимость с glibc (иначе protoc может не запуститься)
RUN apk add --no-cache bash libc6-compat

WORKDIR /home/gradle/project

# Копируем только gradle скрипты и proto для кеша
COPY java-backend/build.gradle java-backend/settings.gradle ./java-backend/
COPY proto ./proto

# Загружаем зависимости и генерим proto (попытаемся использовать кеш)
WORKDIR /home/gradle/project/java-backend
RUN gradle --no-daemon generateProto || true

# Копируем исходники
WORKDIR /home/gradle/project
COPY java-backend ./java-backend

# Финальная сборка fat-jar
WORKDIR /home/gradle/project/java-backend
RUN gradle --no-daemon clean bootJar

FROM eclipse-temurin:21-jre
WORKDIR /app
COPY --from=builder /home/gradle/project/java-backend/build/libs/*.jar /app/app.jar
EXPOSE 8080 6566
ENV JAVA_OPTS=""
ENTRYPOINT ["sh","-c","java $JAVA_OPTS -jar /app/app.jar"]


