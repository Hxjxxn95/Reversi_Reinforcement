import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ReversiEnv(gym.Env):
    metadata = {"render_modes": ["text"], "render_fps": 4}

    def __init__(self, render_mode=None, size=8):
        self.size = size  # The size of the square grid

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Observation: for each cell in size x size matrix, three states involving  
        # { empty (0), agent player 1 (1), gym layer 2 (2) }
        self.observation_space = spaces.Box(low=0, high=255, shape=(size, size, 1), dtype=np.uint8)

        # We have size x size actions, (row, column) index to shoot
        self.action_space = spaces.MultiDiscrete([size, size])

    # Defines all possible movements on the board.
    def possible_moves(self, player, state):
        ret = list()
        opponent = 3 - player

        for i in range(8):
            for j in range(8):
                # if occupied, skip
                if state[i, j] > 0:
                    continue

                # if there is any adjacent cell occupied by oppenent
                # (u, v) is the relative position of eight adjacent cells
                # and also the direction to examine if any opponent piece
                # flipped over
                # (u, v) is
                # (-1, -1) ( -1,  0) (-1,  1)
                # ( 0, -1) (  0,  0) ( 0,  1)
                # ( 1, -1) (  1,  0) ( 1,  1)
                # e.g.,                  
                #          (i-1, j+1) -> opponent
                #    (i, j) -> player
                # -> examine diagonal direction of up and right
                #    (i-1,j+1), (i-2,j+2), (i-3,j+3), ...
                for u in [-1, 0, 1]:
                    for v in [-1, 0, 1]:
                        if (u != 0 or v != 0) and \
                            (i+u>=0 and i+u<8) and \
                                (j+v>=0 and j+v<8) and \
                                    state[i+u,j+v] == opponent:
                            
                            # examine if there is a player's piece
                            # and thus the opponent's pieces between
                            # (i,j) and (i+d*u,j+d*v) can be
                            # flipped over
                            for d in range(1,8):
                                if (i+d*u<0 or i+d*u>=8) or \
                                    (j+d*v<0 or j+d*v>=8):
                                    break

                                if state[i+d*u,j+d*v] == player:
                                    ret.append((i, j))
                                elif state[i+d*u,j+d*v] == opponent:
                                    continue
                                else:
                                    break
        return list(dict.fromkeys(ret))
    
    def scoring(self, _state, _player):
        _enemy = 3 - _player
        ptr1 = np.sum([
            1 if _state[i, j] == _player else 0
            for i in range(8) for j in range(8) ])
        
        ptr2 = np.sum([
            1 if _state[i, j] == _enemy else 0
            for i in range(8) for j in range(8) ])
        
        return ptr1 - ptr2
    
    # flip pieces between the new piece placed and the piece with the same color
    # vertically, horizontally and diagonally
    def flip_piece(self, _state, _pos, _player):
        # assume that the cell of pos is empty
        assert _state[_pos] == 0
        _state[_pos] = _player

        n_flipped = 0

        # find target vertical toward top
        for i in range (-1, -8, -1):
            # out of board
            if _pos[0] + i < 0:
                break

            # if empty, stop proving
            if _state[_pos[0] + i, _pos[1]] == 0:
                break

            # found
            if _state[_pos[0] + i, _pos[1]] == _player:
                for j in range (-1, i, -1):
                    _state[_pos[0] + j, _pos[1]] = _player
                    n_flipped += 1
                break

        # find target vertical toward bottom
        for i in range (1, 8):
            # out of board
            if _pos[0] + i >= 8:
                break

            # if empty, stop proving
            if _state[_pos[0] + i, _pos[1]] == 0:
                break

            # found
            if _state[_pos[0] + i, _pos[1]] == _player:
                for j in range (1, i):
                    _state[_pos[0] + j, _pos[1]] = _player
                    n_flipped += 1
                break

        # find target horizon toward left
        for i in range (-1, -8, -1):
            # out of board
            if _pos[1] + i < 0:
                break

            # if empty, stop proving
            if _state[_pos[0], _pos[1] + i] == 0:
                break

            # found
            if _state[_pos[0], _pos[1] + i] == _player:
                for j in range (-1, i, -1):
                    _state[_pos[0], _pos[1] + j] = _player
                    n_flipped += 1
                break

        # find target horizon toward right
        for i in range (1, 8):
            # out of board
            if _pos[1] + i >= 8:
                break

            # if empty, stop proving
            if _state[_pos[0], _pos[1] + i] == 0:
                break

            # found
            if _state[_pos[0], _pos[1] + i] == _player:
                for j in range (1, i):
                    _state[_pos[0], _pos[1] + j] = _player
                    n_flipped += 1
                break

        # find target diagonal toward top & left
        for i in range (-1, -8, -1):
            # out of board
            if _pos[0] + i < 0 or _pos[1] + i < 0:
                break

            # if empty, stop proving
            if _state[_pos[0] + i, _pos[1] + i] == 0:
                break

            # found
            if _state[_pos[0] + i, _pos[1] + i] == _player:
                for j in range (-1, i, -1):
                    _state[_pos[0] + j, _pos[1] + j] = _player
                    n_flipped += 1
                break

        # find target diagonal toward bottom & right
        for i in range (1, 8):
            # out of board
            if _pos[0] + i >= 8 or _pos[1] + i >= 8:
                break

            # if empty, stop proving
            if _state[_pos[0] + i, _pos[1] + i] == 0:
                break

            # found
            if _state[_pos[0] + i, _pos[1] + i] == _player:
                for j in range (1, i):
                    _state[_pos[0] + j, _pos[1] + j] = _player
                    n_flipped += 1
                break

        # find target diagonal toward top & right
        for i in range (-1, -8, -1):
            # out of board
            if _pos[0] + i < 0 or _pos[1] - i >= 8:
                break

            # if empty, stop proving
            if _state[_pos[0] + i, _pos[1] - i] == 0:
                break

            # found
            if _state[_pos[0] + i, _pos[1] - i] == _player:
                for j in range (-1, i, -1):
                    _state[_pos[0] + j, _pos[1] - j] = _player
                    n_flipped += 1
                break

        # find target diagonal toward bottom & right
        for i in range (1, 8):
            # out of board
            if _pos[0] + i >= 8 or _pos[1] - i < 0:
                break

            # if empty, stop proving
            if _state[_pos[0] + i, _pos[1] - i] == 0:
                break

            # found
            if _state[_pos[0] + i, _pos[1] - i] == _player:
                for j in range (1, i):
                    _state[_pos[0] + j, _pos[1] - j] = _player
                    n_flipped += 1
                break

        return n_flipped
        
    def _get_obs(self):
        return self.state

    def _get_info(self):
        action_mask = np.zeros(shape=(self.size,self.size),dtype=np.int32)
        for idx in self.possible_moves(self.AGENT_PLAYER, self.state):
            action_mask[idx] = 1
        return {'action_mask': action_mask}

    def reset(self, seed=None, options={'gym_player_id': 2}):
        # Define the board to use for the game.
        self.state = np.array([
            [  0,  0,  0,  0,  0,  0,  0,  0 ],
            [  0,  0,  0,  0,  0,  0,  0,  0 ],
            [  0,  0,  0,  0,  0,  0,  0,  0 ],
            [  0,  0,  0,  1,  2,  0,  0,  0 ],
            [  0,  0,  0,  2,  1,  0,  0,  0 ],
            [  0,  0,  0,  0,  0,  0,  0,  0 ],
            [  0,  0,  0,  0,  0,  0,  0,  0 ],
            [  0,  0,  0,  0,  0,  0,  0,  0 ]
        ], dtype=np.int16)

        # if env takes the first move
        if options['gym_player_id'] == 1:
            self.AGENT_PLAYER = 2
            self.GYM_PLAYER = 1
            self.step(self.choose_action())
        else:
            self.AGENT_PLAYER = 1
            self.GYM_PLAYER = 2

        return self._get_obs(), self._get_info()

    # select & return a random action
    def choose_action(self):
        moves = self.possible_moves(self.GYM_PLAYER, self.state)
        return moves[np.random.randint(low=0, high=len(moves))]
    
    def is_over(self):
        # if no more empty cell
        if self.state.min() > 0:
            return True
        
        # if no more possible moves for both players
        if len(self.possible_moves(self.GYM_PLAYER, self.state)) == 0 and \
                len(self.possible_moves(self.AGENT_PLAYER, self.state)) == 0:
            return True
        return False

    def step(self, action):
        assert 0 <= action[0] < self.size and 0 <= action[1] < self.size, "Invalid action: out of bounds"
        
        possible_moves = self.possible_moves(self.AGENT_PLAYER, self.state)
        assert possible_moves, "No possible moves for the agent"
        assert action in possible_moves, f"Invalid action {action}. Possible moves are {possible_moves}"

        # 에이전트 행동 전 상태 저장
        state_before = self.state.copy()

        # 에이전트 플레이어의 행동 실행
        flipped_by_agent = self.flip_piece(self.state, action, self.AGENT_PLAYER)
        assert flipped_by_agent > 0, "The move must flip at least one opponent's piece"

        # 즉각적인 보상 계산
        immediate_reward = flipped_by_agent

        # 전략적 위치에 대한 보상
        if action in [(0,0), (0,7), (7,0), (7,7)]:  # 코너
            immediate_reward += 5
        elif action[0] in [0, 7] or action[1] in [0, 7]:  # 가장자리
            immediate_reward += 2

        # 게임 종료 여부 확인
        terminated = self.is_over()

        if not terminated:
            # 환경(GYM) 플레이어의 턴
            gym_moves = self.possible_moves(self.GYM_PLAYER, self.state)
            if gym_moves:
                gym_action = self.choose_action()
                flipped_by_gym = self.flip_piece(self.state, gym_action, self.GYM_PLAYER)
                # 상대방에 의해 뒤집힌 돌에 대한 페널티
                immediate_reward -= flipped_by_gym * 1.5  # 페널티를 1.5배로 설정
            
            terminated = self.is_over()
        
        # 최종 보상 계산
        if terminated:
            final_score = self.scoring(self.state, self.AGENT_PLAYER)
            if final_score > 0:
                delayed_reward = 100  # 승리 시 큰 양의 보상
            elif final_score < 0:
                delayed_reward = -150  # 패배 시 더 큰 음의 보상
            else:
                delayed_reward = 0  # 무승부
        else:
            delayed_reward = 0

        # 전체 보상 계산
        total_reward = immediate_reward + delayed_reward

        # 게임 진행 상황 (빈 칸의 수로 추정)
        empty_cells = np.sum(self.state == 0)
        game_progress = 1 - (empty_cells / (self.size * self.size))

        observation = self._get_obs()
        info = self._get_info()

        return observation, total_reward, terminated, False, info

    def render(self):
        if self.render_mode == "text":
            return self._render_board()

    def _render_board(self):
        print("Current state of the board:")
        print("    0 1 2 3 4 5 6 7")
        print("-------------------")
        for j in range(8):
            print('{} | {}'.format(j, ' '.join([".12"[self.state[j, i]] for i in range(8)])))

        if self.is_over():
            score = self.scoring(self.state, self.AGENT_PLAYER)
            if score > 0:
                print("Game over! - Winner {}".format(self.AGENT_PLAYER))
            elif score  < 0:
                print("Game over! - Winner {}".format(3 - self.AGENT_PLAYER))
            else:
                print("Tie")

        print(f"Gym player: {self.GYM_PLAYER}, Agent player: {self.AGENT_PLAYER}")
        print(str)
