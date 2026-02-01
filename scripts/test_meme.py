#!/usr/bin/env python3
"""
Test script for meme generation and playback.

Modes:
- compose: Generate a meme from template with text
- play: Display a meme with audio
- demo: Full demo with random templates and sounds
- list: List available templates and sounds
"""

import argparse
import random
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.data import TemplateManager, SoundManager
from src.meme import MemeComposer, MemePlayer, MemeConfig, PlayerConfig


def setup_logging(verbose: bool) -> None:
    """Configure logging level."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, format="<level>{message}</level>")


def list_resources(templates_dir: str, sounds_dir: str) -> None:
    """List available templates and sounds."""
    print("\n=== Available Resources ===\n")
    
    # Templates
    tm = TemplateManager(templates_dir)
    templates = tm.templates
    print(f"Templates ({len(templates)}):")
    for i, t in enumerate(templates[:10]):
        print(f"  {i+1}. {t.name} ({t.width}x{t.height})")
    if len(templates) > 10:
        print(f"  ... and {len(templates) - 10} more")
    
    print()
    
    # Sounds
    sm = SoundManager(sounds_dir)
    sounds = sm.available_sounds
    print(f"Sounds ({len(sounds)} available, {len(sm)} total):")
    for i, s in enumerate(sounds[:10]):
        print(f"  {i+1}. {s.name}")
    if len(sounds) > 10:
        print(f"  ... and {len(sounds) - 10} more")
    
    print()


def compose_meme(
    templates_dir: str,
    output_path: str,
    top_text: str,
    bottom_text: str,
    template_index: int = -1
) -> None:
    """Compose a meme and save to file."""
    tm = TemplateManager(templates_dir)
    composer = MemeComposer()
    
    if not tm.templates:
        logger.error("No templates found!")
        return
    
    # Select template
    if template_index >= 0 and template_index < len(tm.templates):
        template = tm.templates[template_index]
    else:
        template = random.choice(tm.templates)
    
    logger.info(f"Using template: {template.name}")
    
    # Compose meme
    meme = composer.compose(
        template=template,
        top_text=top_text,
        bottom_text=bottom_text
    )
    
    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    meme.save(output)
    logger.info(f"Saved meme to: {output}")


def play_meme(
    templates_dir: str,
    sounds_dir: str,
    template_index: int = -1,
    sound_index: int = -1,
    duration: float = 5.0
) -> None:
    """Display a meme with audio."""
    tm = TemplateManager(templates_dir)
    sm = SoundManager(sounds_dir)
    composer = MemeComposer()
    
    if not tm.templates:
        logger.error("No templates found!")
        return
    
    # Select template
    if template_index >= 0 and template_index < len(tm.templates):
        template = tm.templates[template_index]
    else:
        template = random.choice(tm.templates)
    
    # Select sound
    sound_path = None
    available_sounds = sm.available_sounds
    if available_sounds:
        if sound_index >= 0 and sound_index < len(available_sounds):
            sound = available_sounds[sound_index]
        else:
            sound = random.choice(available_sounds)
        sound_path = sound.path
        logger.info(f"Using sound: {sound.name}")
    else:
        logger.warning("No sounds available")
    
    logger.info(f"Using template: {template.name}")
    
    # Compose with random text
    meme = composer.compose_random(
        template=template,
        use_top_text=True,
        use_bottom_text=True,
        sound_name=sound.name if sound_path else None
    )
    
    # Play
    config = PlayerConfig(display_duration=duration)
    with MemePlayer(config) as player:
        player.show_message("Press ESC to skip, Q to quit", duration=1.5)
        player.display(meme, sound_path=sound_path)


def run_demo(
    templates_dir: str,
    sounds_dir: str,
    num_memes: int = 5,
    duration: float = 5.0
) -> None:
    """Run a demo showing multiple memes."""
    tm = TemplateManager(templates_dir)
    sm = SoundManager(sounds_dir)
    composer = MemeComposer()
    
    if not tm.templates:
        logger.error("No templates found!")
        return
    
    templates = tm.get_random(num_memes)
    sounds = sm.get_random(num_memes) if sm.available_sounds else []
    
    logger.info(f"Running demo with {len(templates)} memes")
    
    config = PlayerConfig(display_duration=duration)
    with MemePlayer(config) as player:
        player.show_message(f"Meme Demo - {num_memes} memes", duration=2.0)
        player.show_countdown(3, "Get ready to laugh!")
        
        for i, template in enumerate(templates):
            if player.should_quit:
                break
            
            # Get sound for this meme
            sound_path = None
            if i < len(sounds) and sounds[i].is_available:
                sound_path = sounds[i].path
            
            # Compose meme
            meme = composer.compose_random(
                template=template,
                use_top_text=random.random() > 0.3,
                use_bottom_text=random.random() > 0.3,
                sound_name=sounds[i].name if sound_path else None
            )
            
            logger.info(f"[{i+1}/{len(templates)}] {meme.template_name}")
            
            # Display
            player.display(meme, sound_path=sound_path)
        
        if not player.should_quit:
            player.show_message("Demo complete!", duration=2.0)


def main():
    parser = argparse.ArgumentParser(
        description="Test meme generation and playback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available resources
  python scripts/test_meme.py --mode list
  
  # Compose a meme and save
  python scripts/test_meme.py --mode compose --top "POV" --bottom "You found a meme" -o output.jpg
  
  # Play a random meme with sound
  python scripts/test_meme.py --mode play
  
  # Run demo with 5 memes
  python scripts/test_meme.py --mode demo --num-memes 5
"""
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["compose", "play", "demo", "list"],
        default="demo",
        help="Operation mode (default: demo)"
    )
    parser.add_argument(
        "--templates-dir",
        default="data/templates",
        help="Directory containing meme templates"
    )
    parser.add_argument(
        "--sounds-dir",
        default="data/sounds",
        help="Directory containing sound files"
    )
    parser.add_argument(
        "--output", "-o",
        default="output/meme.jpg",
        help="Output file path for compose mode"
    )
    parser.add_argument(
        "--top",
        default="",
        help="Top text for meme (compose mode)"
    )
    parser.add_argument(
        "--bottom",
        default="",
        help="Bottom text for meme (compose mode)"
    )
    parser.add_argument(
        "--template-index", "-t",
        type=int,
        default=-1,
        help="Template index (-1 for random)"
    )
    parser.add_argument(
        "--sound-index", "-s",
        type=int,
        default=-1,
        help="Sound index (-1 for random)"
    )
    parser.add_argument(
        "--num-memes", "-n",
        type=int,
        default=5,
        help="Number of memes for demo mode"
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=5.0,
        help="Display duration per meme in seconds"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    try:
        if args.mode == "list":
            list_resources(args.templates_dir, args.sounds_dir)
        
        elif args.mode == "compose":
            compose_meme(
                templates_dir=args.templates_dir,
                output_path=args.output,
                top_text=args.top,
                bottom_text=args.bottom,
                template_index=args.template_index
            )
        
        elif args.mode == "play":
            play_meme(
                templates_dir=args.templates_dir,
                sounds_dir=args.sounds_dir,
                template_index=args.template_index,
                sound_index=args.sound_index,
                duration=args.duration
            )
        
        elif args.mode == "demo":
            run_demo(
                templates_dir=args.templates_dir,
                sounds_dir=args.sounds_dir,
                num_memes=args.num_memes,
                duration=args.duration
            )
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            raise


if __name__ == "__main__":
    main()
