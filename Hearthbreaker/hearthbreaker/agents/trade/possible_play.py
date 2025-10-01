from hearthbreaker.agents.trade.util import Util
from functools import reduce


class PossiblePlay:
    def __init__(self, cards, available_mana):
        if len(cards) == 0:
            raise Exception("PossiblePlay cards is empty")

        self.cards = cards
        self.available_mana = available_mana

    def card_mana(self):
        def eff_mana(card):
            if card.name == "The Coin":
                return -1
            else:
                return card.mana_cost()

        return reduce(lambda s, c: s + eff_mana(c), self.cards, 0)

    def sorted_mana(self):
        return Util.reverse_sorted(map(lambda c: c.mana_cost(), self.cards))

    def wasted(self):
        return self.available_mana - self.card_mana()

    def value(self):
        res = self.card_mana()
        wasted = self.wasted()
        if wasted < 0:
            raise Exception("Too Much Mana")

        res += wasted * -100000000000

        factor = 100000000
        for card_mana in self.sorted_mana():
            res += card_mana * factor
            factor = factor / 10

        if self.has_hero_power() and self.available_mana < 6:
            res -= 10000000000000000

        if any(map(lambda c: c.name == "The Coin", self.cards)):
            res -= 100

        return res

    def has_hero_power(self):
        for card in self.cards:
            if card.name == 'Hero Power':
                return True
        return False

    def first_card(self):
        if self.has_hero_power():
            for card in self.cards:
                if card.name == 'Hero Power':
                    return card
            raise Exception("bad")
        else:
            return self.cards[0]

    def __str__(self):
        names = [c.name for c in self.cards]
        s = str(names)
        return "{} {}".format(s, self.value())


class CoinPlays:
    def coin(self):
        cards = [c for c in filter(lambda c: c.name == 'The Coin', self.cards)]
        return cards[0]

    def raw_plays_with_coin(self):
        res = []
        if self.has_coin():
            coinPlays = self.after_coin().raw_plays()

            for play in coinPlays:
                cards = [self.coin()] + play
                res.append(cards)

        return res

    def raw_plays(self):
        res = []
        for play in self.raw_plays_without_coin():
            res.append(play)

        for play in self.raw_plays_with_coin():
            res.append(play)

        return res

    def has_coin(self):
        return any(map(lambda c: c.name == "The Coin", self.cards))

    def cards_without_coin(self):
        return Util.filter_out_one(self.cards, lambda c: c.name == "The Coin")

    def after_coin(self):
        return PossiblePlays(self.cards_without_coin(), self.mana + 1)

    def without_coin(self):
        return PossiblePlays(self.cards_without_coin(), self.mana)


class HeroPowerCard:
    def __init__(self):
        self.mana = 2
        self.name = "Hero Power"
        self.player = None

    def can_use(self, player, game):
        return player.hero.power.can_use()  #True

    def mana_cost(self):
        return 2


class PossiblePlays(CoinPlays):
    def __init__(self, cards, mana, allow_hero_power=True):
        self.cards = cards
        self.mana = mana
        self.allow_hero_power = allow_hero_power

    def possible_is_pointless_coin(self, possible):
        if len(possible) != 1 or possible[0].name != "The Coin":
            return False

        cards_playable_after_coin = [card for card in filter(lambda c: c.mana - 1 == self.mana, self.cards)]
        return len(cards_playable_after_coin) == 0

    def raw_plays_without_coin(self):
        res = []

        def valid_card(card):
            saved_mana = card.player.mana
            card.player.mana = self.mana
            usable = card.can_use(card.player, card.player.game)
            card.player.mana = saved_mana
            return usable

        possible = [card for card in
                    filter(valid_card, self.cards)]

        if self.possible_is_pointless_coin(possible):
            possible = []

        if self.mana >= 2 and self.allow_hero_power:
            possible.append(HeroPowerCard())

        if len(possible) == 0:
            return [[]]

        for card in possible:
            if card.name == 'Hero Power':
                # For Hero Power, use the original cards without hero power possibility
                f_plays = PossiblePlays(self.cards,  # Use original cards, not rest
                                        self.mana - card.mana_cost(),
                                        allow_hero_power=False).raw_plays()
            else:
                rest = self.cards[:]  # Proper copy
                rest.remove(card)
                f_plays = PossiblePlays(rest,
                                        self.mana - card.mana_cost(),
                                        allow_hero_power=self.allow_hero_power).raw_plays()
            
            for following_play in f_plays:
                combined = [card] + following_play
                res.append(combined)
            #rest = self.cards[0:99999]

            #if card.name == 'Hero Power':
            #    f_plays = PossiblePlays(rest,
            #                            self.mana - card.mana_cost(),
            #                            allow_hero_power=False).raw_plays()
            #else:
            #    rest.remove(card)
            #    f_plays = PossiblePlays(rest,
            #                            self.mana - card.mana_cost(),
            #                            allow_hero_power=self.allow_hero_power).raw_plays()

            #for following_play in f_plays:
            #    combined = [card] + following_play
            #    res.append(combined)

        res = Util.uniq_by_sorted(res)

        return res

    def plays_inner(self):
        res = [PossiblePlay(raw, self.mana) for raw in self.raw_plays() if len(raw) > 0]
        res = sorted(res, key=PossiblePlay.value)
        res.reverse()

        return res

    def plays(self):
        return self.plays_inner()

    def __str__(self):
        res = []
        for play in self.plays():
            res.append(play.__str__())
        return str.join("\n", res)


# + active=""
# class PlayMixin:
#     def play_one_card(self, player):
#         if len(player.minions) == 7:
#             return
#         if player.game.game_ended:
#             return
#
#         allow_hero_power = (not player.hero.power.used) and player.hero.health > 2
#         plays = PossiblePlays(player.hand, player.mana, allow_hero_power=allow_hero_power).plays()
#
#         if len(plays) > 0:
#             play = plays[0]
#             if len(play.cards) == 0:
#                 raise Exception("play has no cards")
#
#             card = play.first_card()
#
#             if card.name == 'Hero Power':
#                 player.hero.power.use()
#             else:
#                 self.last_card_played = card
#                 player.game.play_card(card)
#
#             return card
#             
#     def play_cards(self, player, _recursion_depth=0, _played_cards=[]):
#         """
#         Simple fix: limit recursion depth
#         """
#         if _recursion_depth > 50:  # Reasonable limit for card plays per turn
#             print(f"Warning: Recursion depth limit reached ({_recursion_depth}), stopping card play", _played_cards)
#             return
#         
#         card = self.play_one_card(player)
#         if card:
#             _played_cards.append(card.name)
#             self.play_cards(player, _recursion_depth + 1, _played_cards)
#             
#     #def play_cards(self, player):
#     #    card = self.play_one_card(player)
#     #    if card:
#     #        self.play_cards(player)

# +
DEBUG = False

class PlayMixin:
    def play_one_card(self, player):
        if len(player.minions) == 7:
            return None
        if player.game.game_ended:
            return None

        # Check hero power availability BEFORE creating possible plays
        allow_hero_power = (not player.hero.power.used) and player.hero.health > 2
        
        # Additional validation: if no cards in hand and hero power already used, stop
        if len(player.hand) == 0 and not allow_hero_power:
            return None

        try:
            plays = PossiblePlays(player.hand, player.mana, allow_hero_power=allow_hero_power).plays()
        except RecursionError:
            if DEBUG:
                print("Warning: Recursion detected in mana cost calculation, stopping card play")
            return None

        if len(plays) > 0:
            play = plays[0]
            if len(play.cards) == 0:
                return None  # Don't raise exception, just return None

            card = play.first_card()

            if card.name == 'Hero Power':
                # CRITICAL: Double-check hero power can actually be used
                if not player.hero.power.can_use():
                    if DEBUG:
                        print(f"Warning: Hero power shows as usable but can_use() returns False. used={player.hero.power.used}, mana={player.mana}")
                    return None
                
                # Use hero power and verify it was marked as used
                player.hero.power.use()
                
                # Verify the state changed
                if not player.hero.power.used:
                    if DEBUG:
                        print("ERROR: Hero power was used but 'used' flag is still False!")
                    # Force set it to prevent infinite loop
                    player.hero.power.used = True
                    
                return card
            else:
                # Verify card is actually in hand before trying to play it
                if card not in player.hand:
                    if DEBUG:
                        print(f"Warning: Trying to play {card.name} but it's not in hand!")
                    return None
                
                # Store initial hand size to verify card was removed
                initial_hand_size = len(player.hand)
                
                self.last_card_played = card
                player.game.play_card(card)
                
                # Verify card was actually removed from hand
                if len(player.hand) >= initial_hand_size:
                    if DEBUG:
                        print(f"Warning: Played {card.name} but hand size didn't decrease! ({initial_hand_size} -> {len(player.hand)})")
                
                return card
        
        return None

    def play_cards(self, player, max_cards=20):
        """
        Play cards with robust state validation and safety limits.
        """
        cards_played = 0
        played_card_names = []
        last_mana = player.mana
        last_hand_size = len(player.hand)
        hero_power_uses = 0
        
        while cards_played < max_cards:
            # Safety check: if no progress is being made, stop
            current_mana = player.mana
            current_hand_size = len(player.hand)
            
            card = self.play_one_card(player)
            if not card:
                break
                
            cards_played += 1
            played_card_names.append(card.name)
            
            # Track hero power usage
            if card.name == 'Hero Power':
                hero_power_uses += 1
                # Hero power should only be usable once per turn
                if hero_power_uses > 1:
                    if DEBUG:
                        print(f"ERROR: Hero power used {hero_power_uses} times in one turn!")
                        print(f"Hero power state: used={player.hero.power.used}, can_use={player.hero.power.can_use()}")
                    break
            
            # Verify game state is changing
            new_mana = player.mana
            new_hand_size = len(player.hand)
            
            # If neither mana decreased nor hand size decreased, something is wrong
            if new_mana >= current_mana and new_hand_size >= current_hand_size:
                if DEBUG:
                    print(f"Warning: No progress after playing {card.name}. Mana: {current_mana}->{new_mana}, Hand: {current_hand_size}->{new_hand_size}")
                # Allow one more iteration in case it's a 0-cost card or coin
                if cards_played > 1:  # But stop if this keeps happening
                    break
            
            # Additional safety: detect card repetition
            if cards_played > 10 and played_card_names[-5:].count(card.name) >= 3:
                if DEBUG:
                    print(f"Warning: {card.name} played {played_card_names[-5:].count(card.name)} times in last 5 plays, possible loop")
                break
        
        if cards_played >= max_cards:
            if DEBUG:
                print(f"Warning: Hit maximum card limit ({max_cards})")
                print(f"Cards played: {played_card_names}")
        elif hero_power_uses > 1:
            if DEBUG:
                print(f"Stopped due to multiple hero power usage: {played_card_names}")
        elif cards_played > 0:
            if DEBUG:
                print(f"Normal end after {cards_played} cards: {played_card_names[:10]}{'...' if cards_played > 10 else ''}")
