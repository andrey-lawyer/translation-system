
package com.java_backend.service;

import com.java_backend.grpc.GoTranslatorClient;
import translator.Translation;

import org.springframework.stereotype.Service


@Service
class TranslationService {

    private final GoTranslatorClient goClient;

    TranslationService() {
        String host = System.getenv("GO_TRANSLATOR_HOST") != null ? System.getenv("GO_TRANSLATOR_HOST") : "localhost";
        int port = System.getenv("GO_TRANSLATOR_PORT") != null ? Integer.parseInt(System.getenv("GO_TRANSLATOR_PORT")) : 6565;
        this.goClient = new GoTranslatorClient(host, port);
    }

    Map<String, String> translate(String text, String fromLang, List<String> langs) {
        Translation.TranslateResponse response = goClient.translate(text, fromLang, langs);
        return response.getTranslationsMap();
    }
}