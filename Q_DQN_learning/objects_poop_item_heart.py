import pygame
import random
import os
from game_config import *


# 똥(Poop) 클래스
class Poop:
    def __init__(self):
        # 똥 이미지 로드 및 크기 조정
        self.image = pygame.image.load(os.path.join(IMAGE_PATH, "image_poop.png"))
        self.width = 30
        self.height = 30
        self.image = pygame.transform.scale(self.image, (self.width, self.height))

        # 똥의 초기 위치 설정 (화면 상단 무작위 X축, 화면 밖 Y축)
        self.x = random.randint(0, SCREEN_WIDTH - self.width)
        self.y = -self.height
        # 똥의 초기 낙하 속도 설정
        self.speed = random.randint(3, 7)
        # 충돌 감지를 위한 Rect 객체 생성
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def fall(self, score, speed_multiplier=1.0):
        # 똥의 Y좌표를 업데이트하여 낙하 효과 구현
        # 점수에 따라 속도가 증가하고, speed_multiplier(퀴즈 결과에 따른 속도 효과)에 영향을 받음
        self.y += (self.speed + score / 500) * speed_multiplier
        # Rect 객체의 Y좌표도 업데이트하여 충돌 감지 위치 동기화
        self.rect.y = self.y

        # 똥이 화면 아래로 완전히 벗어나면 다시 화면 상단으로 리셋
        if self.y > SCREEN_HEIGHT:
            self.x = random.randint(
                0, SCREEN_WIDTH - self.width
            )  # 새 X좌표 무작위 설정
            self.y = -self.height  # 화면 밖 상단으로 Y좌표 설정
            self.speed = random.randint(3, 7)  # 속도 재설정
            self.rect.x = self.x  # Rect 객체 X좌표 업데이트
            self.rect.y = self.y  # Rect 객체 Y좌표 업데이트

    def draw(self):
        # 화면에 똥 이미지 그리기
        screen.blit(self.image, (self.x, self.y))


# 아이템(Item) 클래스
class Item:
    def __init__(self):
        # 아이템 종류 무작위 설정 (1이면 쉴드, 2이면 하트, 그 외는 아무 효과 없음)
        self.kind = random.randint(1, 10)
        self.width = 50
        self.height = 50

        # 아이템 이미지 로드 및 크기 조정
        self.image_1 = pygame.image.load(
            os.path.join(IMAGE_PATH, "image_shield.png")
        )  # 쉴드 이미지
        self.image_1 = pygame.transform.scale(self.image_1, (self.width, self.height))
        self.image_2 = pygame.image.load(
            os.path.join(IMAGE_PATH, "image_love.png")
        )  # 하트 이미지
        self.image_2 = pygame.transform.scale(self.image_2, (self.width, self.height))

        # 아이템 초기 위치 설정 (화면 상단 무작위 X축)
        self.x = random.randint(0, SCREEN_WIDTH - self.width)
        self.y = 0
        # 충돌 감지를 위한 Rect 객체 생성
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def fall(self):
        # 아이템의 Y좌표를 업데이트하여 낙하 효과 구현
        self.y += 7
        # Rect 객체의 Y좌표도 업데이트하여 충돌 감지 위치 동기화
        self.rect.y = self.y

        # 아이템이 화면 아래로 완전히 벗어나면 다시 화면 상단으로 리셋
        if self.y >= SCREEN_HEIGHT:
            self.x = random.randint(
                0, SCREEN_WIDTH - self.width
            )  # 새 X좌표 무작위 설정
            self.y = -self.height  # 화면 밖 상단으로 Y좌표 설정
            self.rect.x = self.x  # Rect 객체 X좌표 업데이트
            self.rect.y = self.y  # Rect 객체 Y좌표 업데이트
            self.kind = random.randint(1, 10)  # 아이템 종류 재설정

    def draw(self):
        # 아이템 종류에 따라 적절한 이미지 그리기
        if self.kind == 1:  # 쉴드 아이템일 경우
            screen.blit(self.image_1, (self.x, self.y))
        elif self.kind == 2:  # 하트 아이템일 경우
            screen.blit(self.image_2, (self.x, self.y))

    def use(self):
        # 아이템 종류(kind)를 반환하여 어떤 아이템이 획득되었는지 알려줌
        return self.kind


# 생명(Heart) 클래스 (플레이어의 생명력을 시각적으로 표현)
class Heart:
    def __init__(self):
        # 생명력을 나타내는 리스트 (초기 3개의 하트)
        self.list = [1, 1, 1]
        # 하트 이미지 로드 및 크기 조정
        self.image = pygame.image.load(os.path.join(IMAGE_PATH, "image_love.png"))
        self.width = 60
        self.height = 60
        self.image = pygame.transform.scale(self.image, (self.width, self.height))

    def draw(self):
        # 현재 생명력(self.list의 길이)에 따라 하트 이미지들을 화면에 그리기
        # 화면 우상단부터 왼쪽으로 나열
        for i in range(len(self.list)):  # 수정: range(len(self.list) + 1)에서 +1 제거
            screen.blit(
                self.image, (SCREEN_WIDTH - self.width * (i + 1), 0)
            )  # 수정: 인덱스 i를 (i+1)로 변경하여 맨 오른쪽부터 그리기
