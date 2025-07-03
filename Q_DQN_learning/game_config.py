import pygame
import os

# 화면 크기 설정
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 600
# Pygame 디스플레이 초기화 및 화면 생성
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# 게임 창 제목 설정
pygame.display.set_caption("poop avoidance game")

# 색상 정의 (RGB 값)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

# 게임 내 시간 관련 상수 정의 (프레임 단위로 추정)
QUIZ_INTERVAL = 3600  # 퀴즈가 나타나는 간격
NEXT_QUIZ_DELAY = 120  # 다음 퀴즈까지의 대기 시간
INVULNERABLE_TIME = 100  # 무적 시간
SPEED_EFFECT_TIME = 400  # 속도 효과 지속 시간


# 이미지 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "Images")
BACKGROUND_IMG = "image_backgound.png"
PLAYER_RIGHT_IMG = "image_player_2.png"
PLAYER_LEFT_IMG = "image_player_1.png"
POOP_IMG = "image_poop.png"
SHIELD_IMG = "image_shield.png"
HEART_IMG = "image_love.png"


# 배경 이미지 로드 및 화면 크기에 맞게 스케일 조정
background = pygame.image.load(os.path.join(IMAGE_PATH, BACKGROUND_IMG))
background = pygame.transform.scale(background, (SCREEN_WIDTH, SCREEN_HEIGHT))

# 방어막(쉴드) 상태 아이콘 이미지 로드 및 크기 조정
shield_state = pygame.image.load(os.path.join(IMAGE_PATH, SHIELD_IMG))
shield_state = pygame.transform.scale(shield_state, (20, 20))
