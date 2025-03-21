# -*- coding: utf-8 -*-
''' Implement Mahjong Judger class
'''
from collections import defaultdict
from rlcard.games.mahjong.utils import card_decoding_dict
import numpy as np

class MahjongJudger:
    ''' Determine what cards a player can play
    '''

    def __init__(self, np_random):
        ''' Initilize the Judger class for Mahjong
        '''
        self.np_random = np_random

    @staticmethod
    def judge_pong_gong(dealer, players, last_player):
        ''' Judge which player has pong/gong
        Args:
            dealer (object): The dealer object.
            players (list): List of all players
            last_player (int): The player id of last player

        '''
        last_card = dealer.table[-1]
        last_card_str = last_card.get_str()
        #last_card_value = last_card_str.split("-")[-1]
        #last_card_type = last_card_str.split("-")[0]
        for player in players:
            hand = [card.get_str() for card in player.hand]
            hand_dict = defaultdict(list)
            for card in hand:
                hand_dict[card.split("-")[0]].append(card.split("-")[1])
            #pile = player.pile

            #### SHORT MAHJONG MODIFIED ####
            # # check gong
            # if hand.count(last_card_str) == 3 and last_player != player.player_id:
            #     return 'gong', player, [last_card]*4

            # check pong
            if hand.count(last_card_str) == 2 and last_player != player.player_id:
                return 'pong', player, [last_card]*3
        return False, None, None

    def judge_chow(self, dealer, players, last_player):
        ''' Judge which player has chow
        Args:
            dealer (object): The dealer object.
            players (list): List of all players
            last_player (int): The player id of last player
        '''

        last_card = dealer.table[-1]
        last_card_str = last_card.get_str()
        last_card_type = last_card_str.split("-")[0]
        last_card_index = last_card.index_num
        for player in players:
            #### SHORT MAHJONG MODIFIED ####
            # if last_card_type != "dragons" and last_card_type != "winds" and last_player == player.get_player_id() - 1: ## removed winds
            if last_card_type != "dragons" and last_player == player.get_player_id() - 1:

                # Create 9 dimensional vector where each dimension represent a specific card with the type same as last_card_type
                # Numbers in each dimension represent how many of that card the player has it in hand
                # If the last_card_type is 'characters' for example, and the player has cards: characters_3, characters_6, characters_3,
                # The hand_list vector looks like: [0,0,2,0,0,1,0,0,0]
                hand_list = np.zeros(9)

                for card in player.hand:
                    if card.get_str().split("-")[0] == last_card_type:
                        hand_list[card.index_num] = hand_list[card.index_num]+1

                #pile = player.pile
                #check chow
                test_cases = []
                if last_card_index == 0:
                    if hand_list[last_card_index+1] > 0 and hand_list[last_card_index+2] > 0:
                        test_cases.append([last_card_index+1, last_card_index+2])
                elif last_card_index < 9:
                    if hand_list[last_card_index-2] > 0 and hand_list[last_card_index-1] > 0:
                        test_cases.append([last_card_index-2, last_card_index-1])
                else:
                    if hand_list[last_card_index-1] > 0 and hand_list[last_card_index+1] > 0:
                        test_cases.append([last_card_index-1, last_card_index+1])

                if not test_cases:
                    continue        

                for l in test_cases:
                    cards = []
                    for i in l:
                        for card in player.hand:
                            if card.index_num == i and card.get_str().split("-")[0] == last_card_type:
                                cards.append(card)
                                break
                    cards.append(last_card)
                    return 'chow', player, cards
        return False, None, None


    def get_handcrafted_reward(self, action_val, cur_hand):
       
        decoded_action = card_decoding_dict[action_val]
        # print(decoded_action)
        if decoded_action == "pong":
            return 0.2
        if decoded_action == "chow":
            return 0.2
        if decoded_action == "stand":
            return 0 # not super sure what to do here oops
        
        # otherwise check if the action_val was made a triplet with the hand
        # this assumes that last card in cur_hand was the card dealt by table
        last_dealt = cur_hand[-1].get_str()
        last_type, last_trait = last_dealt.split("-")
        hand = [card.get_str() for card in cur_hand]
        # print(hand)
        _dict = {card: hand.count(card) for card in hand}
        # first check if last card made a triplet
        if _dict[last_dealt] == 3:
            if decoded_action != last_dealt:
                return 0.2
        # next check if last card made a sequence
        possible_sequences = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
        if last_type not in ['dots', 'bamboo', 'characters']:
            return 0
        for sequence in possible_sequences:
            check = [last_type+"-"+str(int(last_trait) + i) for i in sequence]
            if _dict.get(check[0], 0)  > 0 and _dict.get(check[1], 0) > 0 and _dict.get(check[2], 0) > 0:
                if decoded_action not in check:
                    return 0.2
        return 0


    def judge_game(self, game):
        ''' Judge which player has win the game
        Args:
            dealer (object): The dealer object.
            players (list): List of all players
            last_player (int): The player id of last player
        '''
        players_val = []
        win_player = -1
        for player in game.players:
            win, val = self.judge_hu(player)
            players_val.append(val)
            if win:
                win_player = player.player_id
                # print("SOMEONE WON! : ", win_player)
        if win_player != -1 or len(game.dealer.deck) == 0:
            # print(players_val)
            return True, win_player, players_val
        else:
            #player_id = players_val.index(max(players_val))
            return False, win_player, players_val

    def judge_hu(self, player):
        ''' Judge whether the player has won the game
        Args:
            player (object): Target player

        Return:
            Result (bool): Win or not
            Maximum_score (int): Set count score of the player
        '''
        ### TODO: IMPLEMENT THIS
        set_count = len(player.pile)  # starting number of sets shown to the table in pile
        hand = [card.get_str() for card in player.hand]
        count_dict = {card: hand.count(card) for card in hand}

        max_set_count = 0
        # check maximum number of sets possible with each tile that can be used as pair
        for pair_option in count_dict:
            if count_dict[pair_option] >= 2:
                # pop pair from hand and calculate max number of sets
                temp_hand = hand.copy()
                for _ in range(2):
                    temp_hand.pop(temp_hand.index(pair_option))
                temp_set_count, _ = self.cal_set(temp_hand)
                # update max sets
                if temp_set_count + set_count > max_set_count:
                    max_set_count = temp_set_count + set_count
                    if max_set_count >= 2:
                        return True, max_set_count
        
        # check max sets possible with no pair tile (in the case of no winning hand)
        temp_set_count, _ = self.cal_set(hand)
        if temp_set_count + set_count > max_set_count:
            max_set_count = temp_set_count + set_count
        return False, max_set_count


    @staticmethod
    def check_consecutive(_list):
        ''' Check if list is consecutive
        Args:
            _list (list): The target list

        Return:
            Result (bool): consecutive or not
        '''
        l = list(map(int, _list))
        if sorted(l) == list(range(min(l), max(l)+1)):
            return True
        return False

    
    def cal_set(self, cards):
        ''' Calculate the set for given cards
        Args:
            Cards (list): List of cards.

        Return:
            Set_count (int): 
            Sets (list): List of cards that has been popped from user's hand
        '''
        if len(cards) < 3:
            return 0, []
        
        _dict = {card: cards.count(card) for card in cards}
        _dict_by_type = {'dots' : [0]*9, 'bamboo' : [0]*9, 'characters' : [0]*9}
        for card in cards:
            _type, _trait = card.split("-")
            if _type in _dict_by_type:
                _dict_by_type[_type][int(_trait) - 1] += 1
        
        # check first card
        target_card = cards[0]
        target_type, target_trait = target_card.split("-") #[0] <= why's this here 
        max_set_count, max_sets = 0, []
        
        # check triplet (pong)
        if _dict[target_card] >= 3:
            temp_hand = cards.copy()
            for _ in range(3):
                temp_hand.pop(temp_hand.index(target_card))
            set_count, sets = self.cal_set(temp_hand)
            if set_count + 1 > max_set_count:
                max_set_count = set_count + 1
                max_sets = [[target_card]*3] + sets

        # check sequences (chow)
        if target_type in _dict_by_type: # only consider sequences for dots, bamboos, chars
            counts = _dict_by_type[target_type]
            if sum(counts) >= 3:
                possible_sequences = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
                for sequence in possible_sequences:
                    check = [int(target_trait) + i - 1 for i in sequence] # -1 for indices
                    if check[0] >= 0 and check[2] < 9:
                        if counts[check[0]] > 0 and counts[check[1]] > 0 and counts[check[2]] > 0:
                            temp_hand = cards.copy()
                            for i in check:
                                temp_hand.pop(temp_hand.index(target_type + "-"+str(i + 1)))
                            set_count, sets = self.cal_set(temp_hand)
                            if set_count + 1 > max_set_count:
                                max_set_count = set_count + 1
                                max_sets = [target_type + "-"+str(check[0]), 
                                            target_type + "-"+str(check[1]), 
                                            target_type + "-"+str(check[2])] + sets

        # check without first card
        temp_hand = cards.copy()
        temp_hand.pop(temp_hand.index(target_card))
        set_count, sets = self.cal_set(temp_hand)
        if set_count > max_set_count:
            max_set_count = set_count
            max_sets = sets
        return max_set_count, max_sets

        

    

### OLD CODE ###

        # def judge_hu(self, player):
    #     ''' Judge whether the player has won the game
    #     Args:
    #         player (object): Target player

    #     Return:
    #         Result (bool): Win or not
    #         Maximum_score (int): Set count score of the player
    #     '''
    #     set_count = 0
    #     hand = [card.get_str() for card in player.hand]
    #     count_dict = {card: hand.count(card) for card in hand}
    #     set_count = len(player.pile)
    #     #### SHORT MAHJONG MODIFIED ####
    #     if set_count >= 2:  # 4 -> 2
    #         return True, set_count
        
    #     # az: below checks number of possible sets formable in hand where "each" is the pair
    #     used = []
    #     maximum = 0
    #     for each in count_dict:
    #         if each in used:
    #             continue
    #         tmp_set_count = 0
    #         tmp_hand = hand.copy()
    #         if count_dict[each] == 2:
    #             for _ in range(count_dict[each]):
    #                 tmp_hand.pop(tmp_hand.index(each))
    #             tmp_set_count, _set = self.cal_set(tmp_hand)
    #             used.extend(_set)
    #             if tmp_set_count + set_count > maximum:
    #                 maximum = tmp_set_count + set_count
    #             #### SHORT MAHJONG MODIFIED ####
    #             if tmp_set_count + set_count >= 2:  # 4 -> 2
    #                 return True, maximum
    #     return False, maximum



    # def cal_set(self, cards):
    #     ''' Calculate the set for given cards
    #     Args:
    #         Cards (list): List of cards.

    #     Return:
    #         Set_count (int):
    #         Sets (list): List of cards that has been pop from user's hand
    #     '''
    #     tmp_cards = cards.copy()
    #     sets = []
    #     set_count = 0
    #     _dict = {card: tmp_cards.count(card) for card in tmp_cards}
    #     # check pong/gang
    #     for each in _dict:
    #         if _dict[each] == 3 or _dict[each] == 4:
    #             set_count += 1
    #             for _ in range(_dict[each]):
    #                 tmp_cards.pop(tmp_cards.index(each))

    #     # get all of the traits of each type in hand (except dragons and winds)
    #     _dict_by_type = defaultdict(list)
    #     for card in tmp_cards:
    #         _type = card.split("-")[0]
    #         _trait = card.split("-")[1]
    #         if _type == 'dragons' or _type == 'winds':
    #             continue
    #         else:
    #             _dict_by_type[_type].append(_trait)
    #     for _type in _dict_by_type.keys():
    #         values = sorted(_dict_by_type[_type])
    #         if len(values) > 2:
    #             for index, _ in enumerate(values):
    #                 if index == 0:
    #                     test_case = [values[index], values[index+1], values[index+2]]
    #                 elif index == len(values)-1:
    #                     test_case = [values[index-2], values[index-1], values[index]]
    #                 else:
    #                     test_case = [values[index-1], values[index], values[index+1]]
    #                 if self.check_consecutive(test_case):
    #                     set_count += 1
    #                     for each in test_case:
    #                         values.pop(values.index(each))
    #                         c = _type+"-"+str(each)
    #                         sets.append(c)
    #                         if c in tmp_cards:
    #                             tmp_cards.pop(tmp_cards.index(c))
    #     return set_count, sets

#if __name__ == "__main__":
#    judger = MahjongJudger()
#    player = Player(0)
#    card_info = Card.info
#    #print(card_info)
#    player.pile.append([Card(card_info['type'][0], card_info['trait'][0])]*3)
#    #player.hand.extend([Card(card_info['type'][0], card_info['trait'][0])]*2)
#    player.hand.extend([Card(card_info['type'][1], card_info['trait'][1])]*4)
#    player.hand.extend([Card(card_info['type'][2], card_info['trait'][1])]*3)
#    player.hand.extend([Card(card_info['type'][0], card_info['trait'][2])]*3)
#    player.hand.extend([Card(card_info['type'][3], card_info['trait'][9])]*2)
#    #player.hand.extend([Card(card_info['type'][2], card_info['trait'][4])]*1)
#    print([card.get_str() for card in player.hand])
#    print(judger.judge_hu(player))
