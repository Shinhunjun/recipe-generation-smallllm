#!/usr/bin/env python3
"""
Results Merger for Sequential DPO Evaluation

Merges results from sequential persona evaluations into a cumulative report.
This allows running evaluations one persona at a time while maintaining
a unified HTML report with all personas.

Usage:
    python merge_results.py \
        --existing evaluation/reports/detailed_results.json \
        --new evaluation/reports/temp_persona_b/detailed_results.json \
        --personas persona_b_indian_veg \
        --output evaluation/reports
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from report_generator import ReportGenerator


class ResultsMerger:
    """
    Merge evaluation results from multiple runs
    """

    def __init__(self, existing_results_path: str, personas_file: str = None):
        """
        Initialize merger

        Args:
            existing_results_path: Path to existing detailed_results.json
            personas_file: Path to personas.yaml (optional)
        """
        self.existing_path = Path(existing_results_path)

        # Load existing results
        if self.existing_path.exists():
            with open(self.existing_path, encoding='utf-8') as f:
                self.existing_results = json.load(f)
            print(f"✅ Loaded existing results from: {self.existing_path}")
        else:
            # Initialize empty results structure
            self.existing_results = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "personas_evaluated": [],
                    "test_count_per_persona": 0,
                    "evaluators_used": [],
                    "total_tests": 0
                },
                "persona_results": {},
                "summary": {
                    "total_tests": 0,
                    "dpo_wins": 0,
                    "sft_wins": 0,
                    "ties": 0,
                    "dpo_win_rate": 0.0,
                    "sft_win_rate": 0.0,
                    "tie_rate": 0.0,
                    "per_persona": {}
                }
            }
            print(f"⚠️  No existing results found, starting fresh")

        # Load personas if provided
        self.personas = None
        if personas_file:
            import yaml
            with open(personas_file) as f:
                self.personas = yaml.safe_load(f)['personas']

    def merge_with_new_results(
        self,
        new_results_path: str,
        persona_ids: List[str]
    ) -> Dict:
        """
        Merge new persona results into existing results

        Args:
            new_results_path: Path to new detailed_results.json
            persona_ids: List of persona IDs being merged

        Returns:
            Merged results dictionary
        """
        # Load new results
        with open(new_results_path, encoding='utf-8') as f:
            new_results = json.load(f)

        print(f"\n📥 Loading new results from: {new_results_path}")
        print(f"   Personas to merge: {', '.join(persona_ids)}")

        # Merge metadata
        self._merge_metadata(new_results, persona_ids)

        # Merge persona results
        for persona_id in persona_ids:
            if persona_id in new_results["persona_results"]:
                print(f"   ✅ Merging {persona_id}")
                self.existing_results["persona_results"][persona_id] = \
                    new_results["persona_results"][persona_id]
            else:
                print(f"   ⚠️  {persona_id} not found in new results")

        # Recompute summary statistics
        self._recompute_summary()

        return self.existing_results

    def _merge_metadata(self, new_results: Dict, persona_ids: List[str]):
        """Merge metadata from new results"""
        existing_meta = self.existing_results["metadata"]
        new_meta = new_results["metadata"]

        # Update timestamp to latest
        existing_meta["timestamp"] = datetime.now().isoformat()

        # Add new personas to evaluated list
        for persona_id in persona_ids:
            if persona_id not in existing_meta.get("personas_evaluated", []):
                existing_meta.setdefault("personas_evaluated", []).append(persona_id)

        # Merge evaluators used
        new_evaluators = new_meta.get("evaluators_used", [])
        existing_evaluators = existing_meta.get("evaluators_used", [])
        for eval_name in new_evaluators:
            if eval_name not in existing_evaluators:
                existing_evaluators.append(eval_name)
        existing_meta["evaluators_used"] = existing_evaluators

        # Update test counts
        existing_meta["test_count_per_persona"] = new_meta.get("test_count_per_persona", 20)

        # Project ID should remain the same
        if "project_id" in new_meta:
            existing_meta["project_id"] = new_meta["project_id"]

    def _recompute_summary(self):
        """Recompute summary statistics across all personas"""
        total_tests = 0
        total_dpo_wins = 0
        total_sft_wins = 0
        total_ties = 0

        per_persona_stats = {}

        for persona_id, persona_data in self.existing_results["persona_results"].items():
            consensus = persona_data["consensus"]["overall"]

            total_tests += consensus["total_tests"]
            total_dpo_wins += consensus["dpo_wins"]
            total_sft_wins += consensus["sft_wins"]
            total_ties += consensus["ties"]

            per_persona_stats[persona_id] = {
                "name": persona_data["persona_name"],
                "dpo_win_rate": consensus["dpo_win_rate"],
                "dpo_wins": consensus["dpo_wins"],
                "sft_wins": consensus["sft_wins"],
                "ties": consensus["ties"],
                "total_tests": consensus["total_tests"]
            }

        # Update summary
        self.existing_results["summary"] = {
            "total_tests": total_tests,
            "dpo_wins": total_dpo_wins,
            "sft_wins": total_sft_wins,
            "ties": total_ties,
            "dpo_win_rate": total_dpo_wins / total_tests if total_tests > 0 else 0,
            "sft_win_rate": total_sft_wins / total_tests if total_tests > 0 else 0,
            "tie_rate": total_ties / total_tests if total_tests > 0 else 0,
            "per_persona": per_persona_stats
        }

        # Update total_tests in metadata
        self.existing_results["metadata"]["total_tests"] = total_tests

        print(f"\n📊 Recomputed Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   DPO Wins: {total_dpo_wins} ({total_dpo_wins/total_tests*100:.1f}%)")
        print(f"   SFT Wins: {total_sft_wins} ({total_sft_wins/total_tests*100:.1f}%)")
        print(f"   Ties: {total_ties} ({total_ties/total_tests*100:.1f}%)")

    def save_results(self, output_dir: str):
        """
        Save merged results

        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        detailed_path = output_path / "detailed_results.json"
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(self.existing_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved detailed results to: {detailed_path}")

        # Save summary
        summary_path = output_path / "summary_stats.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(self.existing_results["summary"], f, indent=2, ensure_ascii=False)
        print(f"💾 Saved summary to: {summary_path}")

    def regenerate_html_report(self, output_dir: str):
        """
        Regenerate HTML report with merged results

        Args:
            output_dir: Directory to save HTML report
        """
        if not self.personas:
            print("⚠️  Cannot generate HTML report: personas not loaded")
            print("   Provide --personas_file to enable HTML report generation")
            return

        print(f"\n📄 Generating HTML report...")
        report_gen = ReportGenerator(self.existing_results, self.personas)
        html_path = Path(output_dir) / "evaluation_report.html"
        report_gen.generate_html_report(str(html_path))
        print(f"💾 Saved HTML report to: {html_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge DPO evaluation results from sequential runs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--existing",
        required=True,
        help="Path to existing detailed_results.json"
    )
    parser.add_argument(
        "--new",
        required=True,
        help="Path to new detailed_results.json to merge"
    )
    parser.add_argument(
        "--personas",
        required=True,
        help="Comma-separated persona IDs being merged"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for merged results"
    )
    parser.add_argument(
        "--personas_file",
        default="data_pipeline/05_dpo_training/personas.yaml",
        help="Path to personas.yaml (for HTML report generation)"
    )

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    personas_file = project_root / args.personas_file

    # Parse persona IDs
    persona_ids = [p.strip() for p in args.personas.split(",")]

    print("="*70)
    print("🔀 DPO Results Merger")
    print("="*70)

    # Initialize merger
    merger = ResultsMerger(
        existing_results_path=args.existing,
        personas_file=str(personas_file) if personas_file.exists() else None
    )

    # Merge new results
    merged_results = merger.merge_with_new_results(
        new_results_path=args.new,
        persona_ids=persona_ids
    )

    # Save merged results
    merger.save_results(args.output)

    # Regenerate HTML report
    merger.regenerate_html_report(args.output)

    print("\n" + "="*70)
    print("✅ Merge Complete!")
    print("="*70)
    print(f"\nMerged {len(persona_ids)} persona(s)")
    print(f"Total personas in results: {len(merged_results['persona_results'])}")
    print(f"Total tests: {merged_results['summary']['total_tests']}")
    print()


if __name__ == "__main__":
    main()
