"""
main.py — Training & Evaluation Entry Point
=============================================

Command-line interface for running the poker RL system.

Commands:
---------
    train        Run PPO self-play training
    eval         Evaluate a saved model vs random/rule-based opponents
    simulate     Run N hands with chosen agents and print statistics
    test         Run the full test suite

Usage Examples:
---------------
    # Run training from scratch:
    python main.py train

    # Resume training from checkpoint:
    python main.py train --checkpoint checkpoints/model_epoch_100.pt

    # Evaluate a model:
    python main.py eval --checkpoint checkpoints/model_epoch_200.pt --hands 500

    # Simulate 10 hands with verbose output:
    python main.py simulate --hands 10 --verbose

    # Run tests:
    python main.py test
"""

import argparse
import sys
import os

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_train(args):
    from training.ppo_trainer import PPOTrainer, PPOConfig
    from model.transformer import TransformerConfig

    cfg = PPOConfig(
        n_players=args.players,
        total_epochs=args.epochs,
        hands_per_rollout=args.rollout_hands,
        checkpoint_dir=args.checkpoint_dir,
    )
    model_cfg = TransformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )

    trainer = PPOTrainer(cfg, model_cfg)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    trainer.train()


def cmd_eval(args):
    import torch
    from model.transformer import PokerTransformer, TransformerConfig
    from model.tokenizer import PokerTokenizer
    from utils.agents import RLAgent, RandomAgent, RuleBasedAgent
    from engine.player import Player
    from engine.game import Game

    if not args.checkpoint:
        print("[eval] --checkpoint required")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = PokerTokenizer()
    cfg = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        num_actions=tokenizer.num_actions,
    )
    model = PokerTransformer(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rl_agent = RLAgent(model, tokenizer, player_id=0, device=device, explore=False)

    players = [Player(i, f"P{i}", 1000) for i in range(args.players)]
    agents = {0: rl_agent}
    for i in range(1, args.players):
        agents[i] = RuleBasedAgent(aggression=0.5)

    game = Game(players, agents, small_blind=10, big_blind=20)

    total_profit = 0
    for hand_num in range(args.hands):
        for p in players:
            if p.chips < 40:
                p.chips = 1000
        chips_before = players[0].chips
        result = game.play_hand()
        profit = players[0].chips - chips_before
        total_profit += profit
        if args.verbose:
            print(f"Hand {hand_num+1}: profit={profit:+d}, stack={players[0].chips}")
        game.rotate_dealer()

    bb_per_100 = (total_profit / args.hands) * (100 / 20)
    print(f"\n=== Evaluation Results ===")
    print(f"  Hands played : {args.hands}")
    print(f"  Total profit : {total_profit:+d} chips")
    print(f"  BB/100 hands : {bb_per_100:.2f}")


def cmd_simulate(args):
    from utils.agents import RandomAgent, RuleBasedAgent, CallAgent
    from engine.player import Player
    from engine.game import Game

    players = [Player(i, f"P{i}", 500) for i in range(args.players)]
    agents = {i: RuleBasedAgent(aggression=0.4) for i in range(args.players)}
    game = Game(players, agents, small_blind=10, big_blind=20)

    for hand_num in range(args.hands):
        for p in players:
            if p.chips < 30:
                p.chips = 500
        result = game.play_hand()
        if args.verbose:
            print(f"\n--- Hand {hand_num + 1} ---")
            print(f"  Winners : {result.winners}")
            print(f"  Winnings: {result.winnings}")
            print(f"  Stacks  : {result.final_stacks}")
            if args.show_history:
                print(f"  History : {' '.join(result.history)}")
        game.rotate_dealer()

    print(f"\n[simulate] Ran {args.hands} hands with {args.players} players.")


def cmd_test(args):
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_engine.py", "-v", "--tb=short"],
        cwd=os.path.dirname(__file__) or "."
    )
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Poker RL System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # train
    train_p = sub.add_parser("train", help="Run PPO training")
    train_p.add_argument("--players", type=int, default=6)
    train_p.add_argument("--epochs", type=int, default=500)
    train_p.add_argument("--rollout-hands", type=int, default=200)
    train_p.add_argument("--d-model", type=int, default=128)
    train_p.add_argument("--n-heads", type=int, default=4)
    train_p.add_argument("--n-layers", type=int, default=4)
    train_p.add_argument("--checkpoint", type=str, default=None)
    train_p.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    # eval
    eval_p = sub.add_parser("eval", help="Evaluate a saved model")
    eval_p.add_argument("--checkpoint", type=str, required=True)
    eval_p.add_argument("--players", type=int, default=6)
    eval_p.add_argument("--hands", type=int, default=500)
    eval_p.add_argument("--verbose", action="store_true")

    # simulate
    sim_p = sub.add_parser("simulate", help="Run a simulation with chosen agents")
    sim_p.add_argument("--players", type=int, default=6)
    sim_p.add_argument("--hands", type=int, default=20)
    sim_p.add_argument("--verbose", action="store_true", default=True)
    sim_p.add_argument("--show-history", action="store_true")

    # test
    sub.add_parser("test", help="Run the test suite")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "test":
        cmd_test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
