import numpy as np
from copy import deepcopy

from rlcard.games.mahjong import Dealer
from rlcard.games.mahjong import Player
from rlcard.games.mahjong import Round
from rlcard.games.mahjong import Judger

class MahjongGame:

    def __init__(self, allow_step_back=False):
        '''Initialize the class MajongGame
        '''
        print("In MahjongGame Init")

        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.num_players = 4

    def init_game(self):
        ''' Initialilze the game of Mahjong

        This version supports two-player Mahjong

        Returns:
            (tuple): Tuple containing:

                (dict): The first state of the game
                (int): Current player's id
        '''
        print("In MahjongGame init_game()")

        # Initialize a dealer that can deal cards
        self.dealer = Dealer(self.np_random)

        # Initialize four players to play the game
        self.players = [Player(i, self.np_random) for i in range(self.num_players)]

        self.judger = Judger(self.np_random)
        self.round = Round(self.judger, self.dealer, self.num_players, self.np_random)

        #### SHORT MAHJONG MODIFIED ####
        # Deal 7 cards to each player to prepare for the game 
        for player in self.players:
            self.dealer.deal_cards(player, 7)

        # Save the hisory for stepping back to the last state.
        self.history = []

        self.dealer.deal_cards(self.players[self.round.current_player], 1)
        state = self.get_state(self.round.current_player)
        self.cur_state = state

        print("INITIAL GAME STATE: ")
        self.print_game_state()

        return state, self.round.current_player

    def step(self, action):
        ''' Get the next state

        Args:
            action (str): a specific action. (call, raise, fold, or check)

        Returns:
            (tuple): Tuple containing:

                (dict): next player's state
                (int): next plater's id
        '''
        # First snapshot the current state
        if self.allow_step_back:
            hist_dealer = deepcopy(self.dealer)
            hist_round = deepcopy(self.round)
            hist_players = deepcopy(self.players)
            self.history.append((hist_dealer, hist_players, hist_round))
        self.round.proceed_round(self.players, action)
        state = self.get_state(self.round.current_player)
        self.cur_state = state

        print("\nNEW GAME STATE: ")
        self.print_game_state()

        return state, self.round.current_player

    def print_game_state(self):
        result = ""

        result += ("=> valid_act: " + str(self.cur_state["valid_act"]) + "\n")
        result += ("=> table: " + ",".join([c.get_str() for c in self.cur_state['table']]) + "\n")
        result += ("=> player: " + str(self.cur_state["player"]) + "\n")
        result += ("=> current_hand: " + ",".join([c.get_str() + " " + str(c.index_num) for c in self.cur_state['current_hand']]) + "\n")
        result += ("=> players_pile: \n")
        for player in self.cur_state["players_pile"]:
            result += ("==> " + str(player) + ": ") + str([[c.get_str() + " " + str(c.index_num) for c in s ] for s in self.cur_state["players_pile"][player]]) + "\n"
        result += ("=> action-cards: " + ",".join([c.get_str() + " " + str(c.index_num) for c in self.cur_state["action_cards"]]))
    
        print(result) 


    def step_back(self):
        ''' Return to the previous state of the game

        Returns:
            (bool): True if the game steps back successfully
        '''
        if not self.history:
            return False
        self.dealer, self.players, self.round = self.history.pop()
        return True

    def get_state(self, player_id):
        ''' Return player's state

        Args:
            player_id (int): player id

        Returns:
            (dict): The state of the player
        '''
        state = self.round.get_state(self.players, player_id)
        return state

    @staticmethod
    def get_legal_actions(state):
        ''' Return the legal actions for current player

        Returns:
            (list): A list of legal actions
        '''
        if state['valid_act'] == ['play']:
            state['valid_act'] = state['action_cards']
            return state['action_cards']
        else:
            return state['valid_act']

    @staticmethod
    def get_num_actions():
        ''' Return the number of applicable actions

        Returns:
            (int): The number of actions. There are 4 actions (call, raise, check and fold)
        '''
        return 38

    def get_num_players(self):
        ''' return the number of players in Mahjong

        returns:
            (int): the number of players in the game
        '''
        return self.num_players

    def get_player_id(self):
        ''' return the id of current player in Mahjong

        returns:
            (int): the number of players in the game
        '''
        return self.round.current_player

    def is_over(self):
        ''' Check if the game is over

        Returns:
            (boolean): True if the game is over
        '''
        win, player, _ = self.judger.judge_game(self)
        self.winner = player
        
        if win:
            print("WINNER: ", self.players[self.winner].get_player_id())
            self.players[self.winner].print_hand()
            self.players[self.winner].print_pile()

        return win
