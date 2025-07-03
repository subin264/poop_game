import pygame
import sys  # sys 모듈을 추가하여 pygame.quit() 후 종료를 명확히 합니다.
from game_config import *
from player import Player
from objects_poop_item_heart import Poop, Item, Heart
from quiz_class import Quiz
from effects import Speed_Effect


def init():
    # 게임에 필요한 전역 변수들을 초기화합니다.
    global score, player, poops, heart, items, invulnerable_timer, quiz, speed_effect, font, clock
    player = Player()  # 플레이어 객체 생성
    heart = Heart()  # 하트(생명) 객체 생성
    poops = [Poop() for _ in range(5)]  # 똥 5개 생성
    clock = pygame.time.Clock()  # 게임 프레임 관리를 위한 Clock 객체 생성
    score = 0  # 점수 초기화
    items = Item()  # 아이템 객체 생성
    font = pygame.font.Font(None, 40)  # 폰트 설정 (기본 폰트, 크기 40)
    quiz = Quiz()  # 퀴즈 객체 생성
    speed_effect = Speed_Effect()  # 속도 효과 객체 생성


# 게임 루프 (실제 게임 진행)
def game_loop():
    global score, player, poops, heart, items, invulnerable_timer, quiz, speed_effect, font
    running = True  # 게임 루프 실행 상태
    score = 0  # 게임 시작 시 점수 초기화
    player = Player()  # 플레이어 재초기화
    poops = [Poop() for _ in range(5)]  # 똥 객체들 재초기화
    items = Item()  # 아이템 재초기화
    invulnerable_timer = 0  # 무적 타이머 초기화
    frame_count = 0  # 게임 프레임 카운터
    font = pygame.font.Font(None, 40)  # 폰트 재설정

    while running:
        # 퀴즈 이벤트 처리
        frame_count += 1
        # 특정 프레임 간격마다 퀴즈 시작 (예: 1800 프레임 = 30초마다)
        if frame_count % QUIZ_INTERVAL == 0:  # game_config.py의 QUIZ_INTERVAL 사용
            quiz.start_quiz(frame_count)

        # 퀴즈 다음 문제 대기 시간 처리
        if quiz.waiting_next and quiz.next_tictoc > 0:
            quiz.next_tictoc -= 1
            if quiz.next_tictoc == 0:
                quiz.next_quiz(frame_count)

        # 퀴즈 시간 초과 처리
        if quiz.is_timeout(frame_count):
            result = False  # 시간 초과는 오답으로 간주
            speed_effect.apply_effect(result)  # 오답에 따른 속도 효과 적용
            quiz.timeout_quiz()  # 퀴즈 시간 초과 처리

        # Pygame 이벤트 처리 (키 입력, 창 닫기 등)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # 창 닫기 버튼 클릭 시
                pygame.quit()  # Pygame 종료
                sys.exit()  # 시스템 종료

            if event.type == pygame.KEYDOWN:  # 키보드 눌림 이벤트
                # 퀴즈 답변 처리
                if (
                    quiz.active and not quiz.waiting_next
                ):  # 퀴즈가 활성화 상태이고 다음 문제 대기 중이 아닐 때만
                    result = None
                    if event.key == pygame.K_o:  # 'O' 키를 눌렀을 때
                        result = quiz.answer_quiz("O")  # 퀴즈에 'O'로 답변
                        if result:  # 정답이면
                            score += (
                                50 * quiz.difficulty_level
                            )  # 점수 획득 (난이도에 따라 증가)
                    elif event.key == pygame.K_x:  # 'X' 키를 눌렀을 때
                        result = quiz.answer_quiz("X")  # 퀴즈에 'X'로 답변
                        if result:  # 정답이면
                            score += (
                                50 * quiz.difficulty_level
                            )  # 점수 획득 (난이도에 따라 증가)

                    if result is not None:  # 답변이 처리되었으면 (O 또는 X를 눌렀으면)
                        speed_effect.apply_effect(
                            result
                        )  # 퀴즈 결과에 따른 속도 효과 적용
                        quiz.Q_show_result(result)  # 정답/오답 결과 화면에 잠시 표시

        # 플레이어 키 입력 처리
        keys = pygame.key.get_pressed()  # 현재 눌린 모든 키 상태 가져오기
        if keys[pygame.K_LEFT]:  # 왼쪽 화살표 키가 눌렸으면
            player.move(-1)  # 플레이어를 왼쪽으로 이동
        if keys[pygame.K_RIGHT]:  # 오른쪽 화살표 키가 눌렸으면
            player.move(1)  # 플레이어를 오른쪽으로 이동

        # 무적 타이머 처리
        if invulnerable_timer > 0:
            invulnerable_timer -= 1  # 타이머 감소
            invulnerable = True  # 무적 상태 활성화
        else:
            invulnerable = False  # 무적 상태 비활성화

        # 속도 효과 업데이트
        speed_effect.update()

        # 아이템, 똥 이동
        for poop in poops:
            poop.fall(
                score, speed_effect.get_multiplier()
            )  # 똥 낙하 (점수 및 속도 효과에 영향 받음)

        items.fall()  # 아이템 낙하

        # 충돌 체크
        for poop in poops:
            if player.rect.colliderect(poop.rect):  # 플레이어와 똥이 충돌했을 때
                if not invulnerable:  # 무적 상태가 아닐 경우
                    player.life -= 1  # 생명 감소

                    if player.life <= 0:  # 생명이 0 이하면 게임 오버
                        game_over_screen()  # 게임 오버 화면으로 전환
                        return  # 게임 루프 종료
                    else:
                        heart.list.pop(0)  # 하트 리스트에서 하나 제거 (화면 표시용)
                # else:
                # invulnerable = False # 무적 상태 중 충돌해도 무적 상태는 유지됨 (원 코드 주석처리된 부분 해석)

                poops.remove(poop)  # 충돌한 똥 제거
                poops.append(Poop())  # 새로운 똥 추가

        if player.rect.colliderect(items.rect):  # 플레이어와 아이템이 충돌했을 때
            item_type = items.use()  # 아이템 사용 (어떤 아이템인지 반환)
            if item_type == 2:  # 하트 아이템일 경우 (예상)
                heart.list.append(1)  # 하트 리스트에 추가 (화면 표시용)
                player.life += 1  # 플레이어 생명 증가

            if item_type == 1:  # 쉴드(무적) 아이템일 경우 (예상)
                invulnerable_timer = (
                    INVULNERABLE_TIME  # 무적 타이머 설정 (game_config.py의 상수 사용)
                )

            items = Item()  # 사용한 아이템 제거 후 새로운 아이템 생성

        score += 1  # 매 프레임마다 점수 증가

        # 그리기 (화면에 요소들을 그림)
        screen.blit(background, (0, 0))  # 배경 이미지 그리기

        if invulnerable:  # 무적 상태일 때 플레이어 주변에 쉴드 효과 그리기
            pygame.draw.ellipse(
                screen,
                (80, 188, 223),  # 파란색 계열
                (
                    player.x - 10,
                    player.y - 10,
                    player.width + 10,
                    player.height + 10,
                ),  # 플레이어 주변에 타원형 그리기
            )

        player.draw()  # 플레이어 그리기
        heart.draw()  # 하트(생명) 그리기
        items.draw()  # 아이템 그리기

        for poop in poops:  # 모든 똥 그리기
            poop.draw()

        # 점수 표시
        score_text = font.render(
            f"SCORE: {score}", True, WHITE
        )  # 점수 텍스트 생성 (흰색)
        screen.blit(score_text, (10, 10))  # 화면 좌상단에 점수 표시
        quiz.draw(screen, font)  # 퀴즈 관련 UI 그리기

        pygame.display.flip()  # 화면 전체 업데이트
        clock.tick(60)  # 초당 60프레임으로 제한

    # 게임 루프가 종료되면 (running = False가 되면) 게임 오버 화면으로 전환
    game_over_screen()


# 게임 오버 화면
def game_over_screen():
    while True:  # 무한 루프 (게임 오버 화면 유지)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # 창 닫기 버튼 클릭 시
                pygame.quit()  # Pygame 종료
                sys.exit()  # 시스템 종료
            if event.type == pygame.KEYDOWN:  # 키보드 눌림 이벤트
                if event.key == pygame.K_SPACE:  # 스페이스 바를 눌렀을 때
                    init()  # 게임 초기화
                    return  # 게임 오버 화면 루프를 빠져나가 game_loop()가 다시 실행되도록 함 (재시작)

        screen.fill(BLACK)  # 화면을 검은색으로 채움
        game_over_text = font.render("GAME OVER!", True, WHITE)  # "GAME OVER!" 텍스트
        score_text = font.render(f"Total pont: {score}", True, WHITE)  # 총 점수 텍스트
        restart_text = font.render(
            "SPACE again game", True, WHITE
        )  # 재시작 안내 텍스트

        # 텍스트들을 화면 중앙에 배치하여 그리기
        screen.blit(
            game_over_text,
            (
                SCREEN_WIDTH // 2 - game_over_text.get_width() // 2,
                SCREEN_HEIGHT // 2 - 50,
            ),
        )
        screen.blit(
            score_text,
            (SCREEN_WIDTH // 2 - score_text.get_width() // 2, SCREEN_HEIGHT // 2),
        )
        screen.blit(
            restart_text,
            (
                SCREEN_WIDTH // 2 - restart_text.get_width() // 2,
                SCREEN_HEIGHT // 2 + 50,
            ),
        )

        pygame.display.flip()  # 화면 업데이트
        clock.tick(60)  # 프레임 제한
