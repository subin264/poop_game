import pygame
import os
from game_config import *


# 플레이어 클래스 정의
class Player:
    def __init__(self):
        # 플레이어 좌우 이동 이미지 로드
        self.image_R = pygame.image.load(
            os.path.join(IMAGE_PATH, "image_player_2.png")
        )  # 오른쪽을 바라보는 이미지
        self.image_L = pygame.image.load(
            os.path.join(IMAGE_PATH, "image_player_1.png")
        )  # 왼쪽을 바라보는 이미지

        # 플레이어 이미지의 너비와 높이 설정
        self.width = 60
        self.height = 80

        # 로드한 이미지를 설정된 너비와 높이로 스케일 조정
        self.image_L = pygame.transform.scale(self.image_L, (self.width, self.height))
        self.image_R = pygame.transform.scale(self.image_R, (self.width, self.height))

        # 플레이어 초기 위치 설정 (화면 하단 중앙)
        self.x = SCREEN_WIDTH // 2 - self.width // 2
        self.y = SCREEN_HEIGHT - self.height - 20

        # 플레이어 현재 방향 ('RIGHT' 또는 'LEFT')
        self.f = "RIGHT"
        # 플레이어 생명력
        self.life = 3
        # 플레이어 이동 속도
        self.speed = 7
        # 플레이어 충돌 감지를 위한 Rect 객체 생성
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def move(self, direction):
        # 플레이어 x 좌표 이동 (direction이 -1이면 왼쪽, 1이면 오른쪽)
        self.x += direction * self.speed
        # Rect 객체의 x 좌표도 업데이트하여 충돌 감지 위치 동기화
        self.rect.x = self.x

        # 플레이어 방향 설정
        if direction < 0:
            self.f = "LEFT"
        else:
            self.f = "RIGHT"

        # 화면 경계 이탈 방지
        if self.x < 0:  # 왼쪽 경계를 넘어가지 않도록
            self.x = 0
            self.rect.x = self.x
        elif self.x > SCREEN_WIDTH - self.width:  # 오른쪽 경계를 넘어가지 않도록
            self.x = SCREEN_WIDTH - self.width
            self.rect.x = self.x

    def draw(self):
        # 플레이어의 현재 방향에 따라 적절한 이미지 그리기
        if self.f == "RIGHT":
            screen.blit(self.image_R, (self.x, self.y))
        if self.f == "LEFT":
            screen.blit(self.image_L, (self.x, self.y))
