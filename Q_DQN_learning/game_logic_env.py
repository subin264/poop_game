import pygame
import sys
import random
import numpy as np
import os

# 기존 게임 모듈 임포트 (screen 객체는 더 이상 game_config에서 가져오지 않음)
from game_config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    WHITE,
    QUIZ_INTERVAL,
    INVULNERABLE_TIME,
    IMAGE_PATH,
)
from player import Player
from objects_poop_item_heart import Poop, Item, Heart
from quiz_class import Quiz
from effects import Speed_Effect


# reinforcement Learning Environment Class (OpenAI Gym style)
class GameEnv:
    def __init__(self, render_mode=True, target_fps=60):
        # Pygame 초기화 (Display 모드는 조건부)
        if not pygame.get_init():  # Pygame이 아직 초기화되지 않았다면 초기화
            pygame.init()

        self.render_mode = render_mode
        self.target_fps = target_fps  # Setting it to 0 means no FPS limit

        # initialize screen and background (depending on render_mode)
        self.screen = None
        self.background = None

        if self.render_mode:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("poop avoidance game")

            # 백그라운드 이미지 로드
            try:
                self.background = pygame.image.load(
                    os.path.join(IMAGE_PATH, "image_background.png")
                )
                self.background = pygame.transform.scale(
                    self.background, (SCREEN_WIDTH, SCREEN_HEIGHT)
                )
            except FileNotFoundError:
                print(
                    f"Error: Background image not found at {os.path.join(IMAGE_PATH, 'image_background.png')}"
                )
                print(
                    "Please ensure 'Images' folder and 'image_background.png' exist in the correct location."
                )
                sys.exit()  # no inamge end 

        # 게임 객체 초기화
        self.player = Player()
        self.heart = Heart()
        self.poops = [Poop() for _ in range(5)]
        self.items = Item()
        self.quiz = Quiz()
        self.speed_effect = Speed_Effect()

        # 게임 상태 변수 초기화
        self.score = 0
        self.frame_count = 0
        self.invulnerable_timer = 0
        self.font = pygame.font.Font(None, 40)
        self.clock = pygame.time.Clock()

        # 행동 공간 정의 및 매핑
        self.action_space = {0: "LEFT", 1: "RIGHT", 2: "STAY", 3: "QUIZ_O", 4: "QUIZ_X"}
        self.n_actions = len(self.action_space)

        # 상태 공간 이산화를 위한 Bin 개수
        self.player_x_bins = 10
        self.poop_x_rel_bins = 10
        self.poop_y_bins = 10
        self.item_x_rel_bins = 10

    def _get_discrete_state(self):
        # 플레이어 X 위치 (0~1 사이로 정규화 후 이산화)
        player_x_norm = self.player.x / SCREEN_WIDTH
        player_x_discrete = min(
            int(player_x_norm * self.player_x_bins), self.player_x_bins - 1
        )

        # find the closest poop (the poop with the smallest Y value above the player)
        closest_poop = None
        min_poop_y = SCREEN_HEIGHT + 100
        for poop in self.poops:
            if poop.y < min_poop_y:
                min_poop_y = poop.y
                closest_poop = poop

        # diffusion of shit information
        poop_x_rel_discrete = 0
        poop_y_discrete = 0
        if closest_poop:
            # normalized and discretized relative X position of the poop (left/right relative to the player)
            poop_x_rel = closest_poop.x - self.player.x
            # discretize the relative X position range as -SCREEN_WIDTH/2 to +SCREEN_WIDTH/2
            poop_x_rel_discrete = min(
                max(
                    int(
                        (poop_x_rel + SCREEN_WIDTH / 2)
                        / SCREEN_WIDTH
                        * self.poop_x_rel_bins
                    ),
                    0,
                ),
                self.poop_x_rel_bins - 1,
            )
            # poop Y 위치 (0~1 사이로 정규화 후 이산화)
            poop_y_discrete = min(
                int(closest_poop.y / SCREEN_HEIGHT * self.poop_y_bins),
                self.poop_y_bins - 1,
            )

        # item 정보 이산화
        item_x_rel_discrete = 0
        item_kind = (
            0  # 0: no item, 1: shield, 2: heart (defined in objects_poop_item_heart.py)
        )
        if self.items.y < SCREEN_HEIGHT:  # 아이템이 화면 내에 있을 때만 정보 포함
            item_x_rel = self.items.x - self.player.x
            item_x_rel_discrete = min(
                max(
                    int(
                        (item_x_rel + SCREEN_WIDTH / 2)
                        / SCREEN_WIDTH
                        * self.item_x_rel_bins
                    ),
                    0,
                ),
                self.item_x_rel_bins - 1,
            )
            item_kind = self.items.kind

        # vtality is already discrete, so it doesn't need to be changed
        life_val = self.player.life
        # Quiz is active (True/False -> 1/0)
        quiz_active_val = 1 if self.quiz.active else 0
        # speed ​​effect multiplier (0.7, 1.0, 1.3 -> 0, 1, 2 등으로 이산화)
        speed_effect_discrete = 1  # default 1.0
        if self.speed_effect.multiplier == 0.7:
            speed_effect_discrete = 0
        elif self.speed_effect.multiplier == 1.3:
            speed_effect_discrete = 2

        # return a state tuple (so it can be used as a key in a Q-table)
        state = (
            player_x_discrete,
            poop_x_rel_discrete,
            poop_y_discrete,
            item_x_rel_discrete,
            item_kind,
            life_val,
            quiz_active_val,
            speed_effect_discrete,
        )
        return state

    def reset(self):
        """ Resets the environment to its initial state and returns the initial state."""
        # 게임 객체 및 상태 변수 초기화
        self.player = Player()
        self.heart = Heart()
        self.poops = [Poop() for _ in range(5)]
        self.items = Item()
        self.quiz = Quiz()
        self.speed_effect = Speed_Effect()

        self.score = 0
        self.frame_count = 0
        self.invulnerable_timer = 0

        # return initial state
        return self._get_discrete_state()

    def step(self, action_idx):
        """
        The environment advances one step based on the agent's actions,
        Returns the next state, reward, and whether the game is over.
        """
        reward = 0.0  # reset the rewards you get in this step
        done = False  # Whether the game is over (True/False)

        # Pygame event handling (only needed when in rendering mode)
        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()  # erminating Pygame and handling sys.exit()
                    # sys.exit() is called in GameEnv.close(), so don't call it again here.
                    return (
                        self._get_discrete_state(),
                        reward,
                        True,
                        {},
                    )  # 즉시 종료 상태 반환

        self.frame_count += 1  # 프레임 카운트 증가

        # 행동 매핑 및 적용
        action = self.action_space[action_idx]
        quiz_action_processed = False  # whether the quiz action was processed in this step

        # 퀴즈 활성화 상태일 때 퀴즈 답변 행동 처리
        if self.quiz.active and not self.quiz.waiting_next:
            if action == "QUIZ_O":
                is_correct = self.quiz.answer_quiz("O")
                self.speed_effect.apply_effect(is_correct)
                self.quiz.Q_show_result(is_correct)
                quiz_action_processed = True
                if is_correct:
                    reward += 15.0  # quiz Correct Answer Reward
                    self.score += 50 * self.quiz.difficulty_level  # 게임 점수 증가
                else:
                    reward -= 15.0  # quiz wrong answer penalty
            elif action == "QUIZ_X":
                is_correct = self.quiz.answer_quiz("X")
                self.speed_effect.apply_effect(is_correct)
                self.quiz.Q_show_result(is_correct)
                quiz_action_processed = True
                if is_correct:
                    reward += 15.0  # 퀴즈 정답 보상
                    self.score += 50 * self.quiz.difficulty_level
                else:
                    reward -= 15.0  # 퀴즈 오답 페널티

        # handle player movement actions only if they are not quiz actions or if the quiz is disabled.
            if action == "LEFT":
                self.player.move(-1)
            elif action == "RIGHT":
                self.player.move(1)
            # "STAY"는 플레이어의 move() 함수를 호출하지 않아 정지 상태 유지

        # 퀴즈 이벤트 및 타이머 업데이트 (매 프레임)
        if (
            self.frame_count % QUIZ_INTERVAL == 0
            and not self.quiz.active
            and not self.quiz.waiting_next
        ):
            self.quiz.start_quiz(self.frame_count)

        if self.quiz.waiting_next and self.quiz.next_tictoc > 0:
            self.quiz.next_tictoc -= 1
            if self.quiz.next_tictoc == 0:
                self.quiz.next_quiz(self.frame_count)

        # 퀴즈 시간 초과 처리
        if self.quiz.is_timeout(self.frame_count):
            if (
                self.quiz.active and not self.quiz.waiting_next
            ):  # 퀴즈가 활성 상태인데 타임아웃된 경우
                self.quiz.timeout_quiz()  # 퀴즈 내부 상태 업데이트
                self.speed_effect.apply_effect(
                    False
                )  # 시간 초과는 오답으로 간주하여 속도 효과 적용
                reward -= 15.0  # 퀴즈 시간 초과 페널티

        # 무적 타이머 업데이트
        if self.invulnerable_timer > 0:
            self.invulnerable_timer -= 1
            invulnerable = True
        else:
            invulnerable = False

        # 속도 효과 업데이트
        self.speed_effect.update()

        # 똥 및 아이템 낙하
        for poop in self.poops:
            poop.fall(self.score, self.speed_effect.get_multiplier())
        self.items.fall()

        # collision check and compensation application
        collided_with_poop = False
        for i, poop in enumerate(self.poops):
            if self.player.rect.colliderect(poop.rect):
                collided_with_poop = True
                if not invulnerable:
                    self.player.life -= 1
                    reward -= 50.0  # 똥 충돌 페널티 (생명 손실 포함)

                    if self.player.life <= 0:
                        done = True  # 게임 종료 조건
                        reward -= 100.0  # 게임 오버 추가 페널티
                    else:
                        if self.heart.list:  # 하트 UI 리스트가 비어있지 않다면 제거
                            self.heart.list.pop(0)

                # 충돌한 똥 제거 및 새로운 똥 추가
                self.poops[i] = (
                    Poop()
                )  # 직접 제거 후 추가 대신, 해당 인덱스에 새 똥 할당 (더 안정적)
                break  # 한 프레임에 여러 똥과 충돌하는 경우 방지

        if self.player.rect.colliderect(self.items.rect):
            item_type = self.items.use()
            if item_type == 2:  # 하트 아이템
                if self.player.life < 3:  # 최대 생명력 이상으로는 증가시키지 않음
                    self.player.life += 1
                    self.heart.list.append(1)
                reward += 30.0  # 하트 획득 보상
            elif item_type == 1:  # 쉴드 아이템
                self.invulnerable_timer = INVULNERABLE_TIME
                reward += 20.0  # 실드 획득 보상

            self.items = Item()  # 사용한 아이템 제거 후 새로운 아이템 생성

        # 생존 보상 (매 프레임마다)
        reward += 0.1

        # 게임 점수 증가 (환경에서는 보상으로 계산하지 않음, 게임 로직 유지)
        self.score += 1

        # 다음 상태 획득
        next_state = self._get_discrete_state()

        # 화면 그리기 (렌더링 모드에 따라)
        if self.render_mode:
            self.render(invulnerable)

        # FPS 제한 적용 (target_fps가 0이 아니면 제한)
        if self.target_fps > 0:
            self.clock.tick(self.target_fps)

        return (
            next_state,
            reward,
            done,
            {},
        )  # 마지막 {}는 추가 정보 딕셔너리 (여기서는 사용 안 함)

    def render(self, invulnerable_status):
        """
        draws the current game state on the Pygame screen.
        It should only be executed when self.render_mode is True.
        """
        if not self.render_mode or self.screen is None:
            return  # 렌더링 모드가 아니거나 화면이 초기화되지 않았으면 아무것도 그리지 않음

        self.screen.blit(self.background, (0, 0))

        if invulnerable_status:
            # 무적 상태일 때 플레이어 주변에 타원형 쉴드 효과 그리기
            pygame.draw.ellipse(
                self.screen,  # self.screen 사용
                (80, 188, 223),  # 파란색 계열
                (
                    self.player.x - 10,
                    self.player.y - 10,
                    self.player.width + 10,
                    self.player.height + 10,
                ),
            )

        # 모든 게임 객체의 draw 메서드에 self.screen을 인자로 전달
        self.player.draw(self.screen)
        self.heart.draw(self.screen)
        self.items.draw(self.screen)

        for poop in self.poops:
            poop.draw(self.screen)

        # 점수 표시
        score_text = self.font.render(f"SCORE: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        # display quiz UI
        self.quiz.draw(self.screen, self.font)

        pygame.display.flip()

    def close(self):
        """
        Clean up Pygame when the environment exits.
        sys.exit() terminates the entire program.
        """
        if pygame.get_init():
            pygame.quit()
