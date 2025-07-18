package com.java_backend.grpc;

import io.grpc.stub.StreamObserver
import org.lognet.springboot.grpc.GRpcService
import translator.Translation
import translator.TranslatorServiceGrpc;
import translator.Translation.TranslateResponse;

@GRpcService
class TranslatorServiceImpl extends TranslatorServiceGrpc.TranslatorServiceImplBase {

    private final GoTranslatorClient goClient = new GoTranslatorClient("localhost", 6565);

    @Override
    void translate(Translation.TranslateRequest request,
                   StreamObserver<TranslateResponse> responseObserver) {

        System.out.println("[TranslatorServiceImpl] Received request from client: " + request);

        try {
            TranslateResponse goResponse = goClient.translate(request.getText(), request.getLangsList());

            System.out.println("[TranslatorServiceImpl] Received response from Go: " + goResponse);

            responseObserver.onNext(goResponse);
            responseObserver.onCompleted();
        } catch (Exception e) {
            System.err.println("[TranslatorServiceImpl] Error while calling Go service: " + e.getMessage());
            responseObserver.onError(e);
        }
    }
}



