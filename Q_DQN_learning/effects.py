from game_config import *


# 퀴즈 결과에 따른 속도 변화 효과를 관리하는 클래스
class Speed_Effect:
    def __init__(self):
        # 현재 속도 배율 (기본값: 1.0, 변화 없음)
        self.multiplier = 1.0
        # 효과 지속 시간 타이머
        self.timer = 0

    def apply_effect(self, is_correct):  # 퀴즈 결과에 따른 속도 효과 적용
        if is_correct:
            self.multiplier = 0.7  # 정답 시 속도 감소 (더 느려짐)
        else:
            self.multiplier = 1.3  # 오답 시 속도 증가 (더 빨라짐)
        self.timer = 400  # 효과 지속 시간을 400프레임으로 설정

    def update(self):  # 매 프레임마다 호출하여 타이머 업데이트
        if self.timer > 0:
            self.timer -= 1  # 타이머 감소
            if self.timer == 0:
                self.multiplier = 1.0  # 타이머가 끝나면 속도 배율을 기본값으로 초기화

    def get_multiplier(self):  # 현재 적용된 속도 배율 반환
        return self.multiplier

    def is_active(self):  # 효과가 활성 상태인지 (타이머가 0보다 큰지) 확인
        return self.timer > 0

    def get_seconds(self):  # 남은 효과 지속 시간(초)을 반환
        # 타이머가 0보다 크면 초 단위로 변환하여 반환 (60프레임 = 1초, 올림 처리)
        return self.timer // 60 + 1 if self.timer > 0 else 0
