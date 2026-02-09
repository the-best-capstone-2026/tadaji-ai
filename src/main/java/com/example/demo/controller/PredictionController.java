package com.example.demo.controller;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
public class PredictionController {

    @PostMapping("/api/predict")
    public String predict(@RequestParam("image") MultipartFile image) {
        // 사진이 잘 들어왔는지 로그로 확인하는 게 1주차 목표예요!
        System.out.println("받은 파일 이름: " + image.getOriginalFilename());
        return "이미지 수신 성공!";
    }
}