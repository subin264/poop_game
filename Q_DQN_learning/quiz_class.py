import pygame
import random
from game_config import *


# 미니게임 시스템: 간단한 수학 문제 퀴즈
class Quiz:
    def __init__(self):
        # 퀴즈 활성화 상태
        self.active = False
        # 현재 문제
        self.question = ""
        # 정답 (O 또는 X)
        self.correct_answer = ""
        # 퀴즈 시작 프레임
        self.start_frame = 0
        # 퀴즈 제한 시간 (프레임 단위) - 900프레임 = 15초 (60프레임/초 기준)
        self.duration_frames = 900
        # 퀴즈 난이도 레벨
        self.difficulty_level = 1
        # 맞힌 문제 수 (누적)
        self.correct_count = 0
        # 한 세트당 퀴즈 문제 수
        self.quiz_set_count = 1
        # 현재 세트에서 맞힌 퀴즈 문제 수
        self.correct_quiz = 0
        # 다음 문제 대기 상태 여부
        self.waiting_next = False
        # 총 시도한 문제 수 (누적)
        self.total_count = 0
        # 다음 문제 대기 시간 (틱톡 타이머)
        self.next_tictoc = 60
        # 마지막 퀴즈 결과 ('Correct!' 또는 'wrong')
        self.last_result = None
        # 결과 표시 타이머
        self.result_timer = 0

    def start_quiz(self, current_frame):
        # 한 세트당 퀴즈 문제 수를 1~3개 사이로 무작위 설정
        self.quiz_set_count = random.randint(1, 3)
        # 현재 세트에서 맞힌 퀴즈 수 초기화
        self.correct_quiz = 0
        # 다음 문제 대기 상태 해제
        self.waiting_next = False
        # 퀴즈 활성화
        self.active = True
        # 퀴즈 시작 프레임 기록
        self.start_frame = current_frame
        self.start_frame = current_frame  # 중복 코드로 보임
        # 새 퀴즈 생성
        self.generate_quiz()

    # 난이도 레벨별 퀴즈 설정 및 생성
    def generate_quiz(self):
        if self.difficulty_level == 1:  # 레벨 1
            num1 = random.randint(1, 25)
            num2 = random.randint(1, 25)
            operation = ["+", "-"]
        elif self.difficulty_level == 2:  # 레벨 2
            num1 = random.randint(10, 120)
            num2 = random.randint(1, 20)
            operation = ["+", "-", "*"]
        elif self.difficulty_level == 3:  # 레벨 3
            num1 = random.randint(20, 250)
            num2 = random.randint(10, 50)
            operation = ["+", "-", "*", "/"]
        else:  # 기본값 (레벨 1)
            self.difficulty_level = 1
            num1 = random.randint(1, 25)
            num2 = random.randint(1, 25)
            operation = ["+", "-"]

        op = random.choice(operation)

        # 나누기 연산을 위해 첫 번째 숫자를 두 번째 숫자의 배수로 만듦 (정수 나눗셈)
        if op == "/":
            num2 = random.randint(2, 15)
            quotient = random.randint(5, 30)
            num1 = num2 * quotient

        # 실제 연산 결과 계산
        if op == "+":
            correct_result = num1 + num2
        elif op == "-":
            # 뺄셈 결과가 음수가 되지 않도록 큰 수에서 작은 수를 뺌
            if num1 < num2:
                num1, num2 = num2, num1
            correct_result = num1 - num2
        elif op == "*":
            correct_result = num1 * num2
        else:  # '/'
            correct_result = num1 // num2  # 정수 나눗셈

        # O/X 문제 생성 (실제 결과와 다르거나 같게 표시)
        if random.choice([True, False]):  # True일 경우 정답과 같은 결과 표시 (정답 'O')
            show_result = correct_result
            self.correct_answer = "O"
        else:  # False일 경우 정답과 다른 결과 표시 (정답 'X')
            # 정답과 오프셋만큼 다른 결과를 생성
            offset = random.randint(1, max(5, correct_result // 10))
            show_result = correct_result + random.choice([-offset, +offset])
            # 우연히 같은 결과가 나오면 1을 더하여 다르게 만듦
            if show_result == correct_result:
                show_result += 1
            self.correct_answer = "X"

        # 질문 문자열 생성
        self.question = f"{num1} {op} {num2} = {show_result}"

    # 퀴즈 답변 처리
    def answer_quiz(self, user_answer):
        # 퀴즈가 비활성화 상태이거나 다음 문제 대기 중이면 처리하지 않음
        if not self.active or self.waiting_next:
            return False

        # 총 시도 문제 수 증가
        self.total_count += 1
        # 사용자 답변이 정답과 일치하는지 확인
        is_correct = user_answer.upper() == self.correct_answer
        if is_correct:
            self.correct_count += 1  # 누적 맞힌 문제 수 증가
        self.correct_quiz += 1  # 현재 세트에서 맞힌 퀴즈 수 증가

        # 다음 문제가 남았으면 다음 문제 대기 상태로 전환 (1초 대기)
        if self.correct_quiz < self.quiz_set_count:
            self.waiting_next = True
            self.active = False
            self.next_tictoc = 60  # 60프레임 (1초) 대기
        else:  # 현재 세트의 모든 퀴즈를 풀었으면 퀴즈 종료 및 레벨 업데이트
            self.active = False
            self.waiting_next = False
            self._update_level()  # 난이도 레벨 업데이트
        return is_correct

    # 정답/오답 결과 표시 (짧은 시간)
    def Q_show_result(self, is_correct):
        self.last_result = "Correct!" if is_correct else "wrong"  # 결과 메시지 설정
        self.result_timer = 60  # 60프레임 (1초) 동안 결과 표시

    # 다음 문제로 넘어가기 전 대기 시간 처리 및 다음 퀴즈 시작
    def next_quiz(self, current_frame):
        # 다음 문제 대기 중이고, 아직 현재 세트의 퀴즈가 남았을 때
        if self.waiting_next and self.correct_quiz < self.quiz_set_count:
            self.waiting_next = False  # 대기 상태 해제
            self.active = True  # 퀴즈 활성화
            self.start_frame = current_frame  # 새 퀴즈 시작 프레임 기록
            self.generate_quiz()  # 새 퀴즈 생성

    # 난이도 레벨 업데이트
    def _update_level(self):
        # 최소 3문제 이상 풀어야 난이도 조정 시작
        if self.total_count < 3:
            return
        accuracy = self.correct_count / self.total_count  # 정확도 계산
        # 정확도가 70% 이상이고 최고 레벨이 아니면 레벨 증가
        if accuracy >= 0.7 and self.difficulty_level < 3:
            self.difficulty_level += 1
        # 정확도가 40% 이하이고 최저 레벨이 아니면 레벨 감소
        elif accuracy <= 0.4 and self.difficulty_level > 1:
            self.difficulty_level -= 1

    # 난이도 레벨에 따른 색상 구분
    def get_difficulty_level_color(self):
        colors = {
            1: (100, 255, 100),  # 레벨 1: 연두색
            2: (255, 255, 100),  # 레벨 2: 노란색
            3: (255, 100, 100),  # 레벨 3: 빨간색
        }
        # 현재 난이도에 맞는 색상 반환, 없으면 흰색 반환
        return colors.get(self.difficulty_level, (255, 255, 255))

    # 시간 초과 여부 확인
    def is_timeout(self, current_frame):
        if not self.active:  # 퀴즈가 활성화 상태가 아니면 시간 초과 없음
            return False
        # 현재 프레임에서 시작 프레임을 빼서 경과 시간을 계산하고, 제한 시간을 초과했는지 확인
        return current_frame - self.start_frame >= self.duration_frames

    # 퀴즈 시간 초과 처리
    def timeout_quiz(self):
        # 퀴즈가 활성화 상태이고 다음 문제 대기 중이 아닐 때만 처리
        if self.active and not self.waiting_next:
            self.total_count += 1  # 총 시도 문제 수 증가 (시간 초과도 오답으로 간주)
            self.correct_quiz += (
                1  # 현재 세트에서 맞힌 퀴즈 수 증가 (다음 퀴즈로 진행 위함)
            )

            # 다음 문제가 남았으면 다음 문제 대기 상태로 전환
            if self.correct_quiz < self.quiz_set_count:
                self.waiting_next = True
                self.active = False
                self.next_tictoc = 60  # 60프레임 (1초) 대기
            else:  # 현재 세트의 모든 퀴즈를 풀었으면 퀴즈 종료 및 레벨 업데이트
                self.active = False
                self.waiting_next = False
                self._update_level()

    # 남은 시간(초) 계산
    def get_seconds(self, current_frame):
        # 퀴즈가 활성화 상태가 아니면 0 반환
        if not self.active:
            return 0
        # 경과 프레임 계산
        ellipsis_frames = current_frame - self.start_frame
        # 남은 프레임 계산 (최소 0)
        seconds_frames = max(0, self.duration_frames - ellipsis_frames)
        # 프레임을 초로 변환하여 반환 (60프레임 = 1초)
        return seconds_frames // 60

    # 정확도 계산
    def get_accuracy(self):
        if self.total_count == 0:  # 시도한 문제가 없으면 0 반환
            return 0
        return (self.correct_count / self.total_count) * 100  # 정확도 (백분율) 반환

    # 화면에 퀴즈 관련 요소 그리기
    def draw(self, screen, font):
        # 퀴즈가 활성화 상태도 아니고 다음 문제 대기 중도 아니면 아무것도 그리지 않음
        if not self.active and not self.waiting_next:
            return

        # 다음 문제 대기 중일 때 오버레이 및 메시지 표시
        if self.waiting_next:
            # 반투명 오버레이 생성 및 그리기
            overlay = pygame.Surface((500, 200))
            overlay.set_alpha(200)  # 투명도 설정
            overlay.fill((255, 255, 255))  # 흰색으로 채우기
            screen.blit(overlay, (150, 200))  # 화면에 오버레이 배치

            # "next coming.." 텍스트 표시
            waiting_text = font.render("next coming..", True, (0, 0, 0))
            text_rect = waiting_text.get_rect(center=(400, 280))
            screen.blit(waiting_text, text_rect)

            # 퀴즈 진행 상황 표시 (예: Quiz 1/3)
            progres_text = font.render(
                f"Quiz{self.correct_quiz + 1}/{self.quiz_set_count}", True, (0, 0, 0)
            )
            screen.blit(progres_text, (500, 200))

            # tictoc (남은 시간) 표시
            seconds = self.next_tictoc // 60 + 1  # 남은 초 계산 (올림)
            tictoc_text = font.render(
                f"{seconds}", True, (250, 0, 0)
            )  # 빨간색으로 표시
            tictoc_rect = tictoc_text.get_rect(center=(400, 320))
            screen.blit(tictoc_text, tictoc_rect)
            return  # 다음 그리기 작업은 건너뜀

        # 결과 메시지 타이머 처리
        if self.result_timer > 0:
            self.result_timer -= 1  # 타이머 감소
        # 결과 메시지가 있고 타이머가 남아있을 때 결과 표시
        if self.result_timer > 0 and self.last_result:
            result_color = (
                (0, 255, 0) if "correct" in self.last_result else (255, 0, 0)
            )  # 결과에 따른 색상 설정
            result_text = font.render(
                self.last_result, True, result_color
            )  # 결과 텍스트 생성
            result_rect = result_text.get_rect(center=(300, 300))  # 위치 설정
            screen.blit(result_text, result_rect)  # 화면에 그리기

        # 질문 텍스트 그리기
        question_text = font.render(
            self.question, True, (0, 0, 0)
        )  # 검은색으로 질문 텍스트 생성
        text_rect = question_text.get_rect(center=(400, 270))  # 중앙 정렬
        screen.blit(question_text, text_rect)  # 화면에 그리기

        # "Press O or X" 안내 텍스트 그리기
        instruction = font.render(
            "Press O or X", True, (0, 0, 0)
        )  # 검은색으로 안내 텍스트 생성
        inst_rect = instruction.get_rect(center=(400, 320))  # 중앙 정렬
        screen.blit(instruction, inst_rect)  # 화면에 그리기

        # 난이도 레벨 텍스트 그리기
        level_text = font.render(f"Level {self.difficulty_level}", True, (0, 0, 0))
        screen.blit(level_text, (170, 220))

        # 퀴즈 진행 상황 텍스트 그리기
        progres_text = font.render(
            f"{self.correct_quiz + 1}/{self.quiz_set_count}", True, (0, 0, 0)
        )
        screen.blit(progres_text, (550, 220))
