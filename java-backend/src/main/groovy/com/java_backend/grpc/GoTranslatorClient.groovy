package com.java_backend.grpc

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import translator.TranslatorServiceGrpc;
import translator.Translation;

class GoTranslatorClient {

    private final TranslatorServiceGrpc.TranslatorServiceBlockingStub stub;

    // Constructor creates a gRPC channel and blocking stub to communicate with Go service
    GoTranslatorClient(String host, int port) {
        ManagedChannel channel = ManagedChannelBuilder
                .forAddress(host, port)
                .usePlaintext()  // Disable TLS for local testing
                .build();

        this.stub = TranslatorServiceGrpc.newBlockingStub(channel);

        System.out.println("[GoTranslatorClient] gRPC channel created: " + host + ":" + port);
    }

    // Sends a translate request to the Go service and returns the response
    Translation.TranslateResponse translate(String text, List<String> langs) {
        System.out.println("[GoTranslatorClient] Sending request to Go service: text = " + text + ", langs = " + langs);

        Translation.TranslateRequest request = Translation.TranslateRequest.newBuilder()
                .setText(text)
                .addAllLangs(langs)
                .build();

        Translation.TranslateResponse response = stub.translate(request);

        System.out.println("[GoTranslatorClient] Received response from Go service: " + response);

        return response;
    }
}
