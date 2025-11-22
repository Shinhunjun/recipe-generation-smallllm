"""
DPO Training Data Validation Script

Validates the quality of DPO training data before model training.
Checks JSON validity, persona constraints, alignment scores, and format correctness.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import re


class DPODataValidator:
    def __init__(self, data_dir: str, personas_config: str):
        self.data_dir = Path(data_dir)
        self.personas_config = personas_config

        # Load personas
        with open(personas_config) as f:
            self.personas = yaml.safe_load(f)['personas']

        # Validation results
        self.results = {
            'critical_errors': [],
            'warnings': [],
            'statistics': defaultdict(dict),
            'violations': defaultdict(list)
        }

    def validate_all(self):
        """Run all validation checks"""
        print("=" * 60)
        print("DPO Training Data Validation")
        print("=" * 60)

        # 1. JSON Validity
        print("\n[1/5] Validating JSON structure...")
        self.validate_json_structure()

        # 2. Persona Constraints
        print("\n[2/5] Checking persona constraints...")
        self.validate_persona_constraints()

        # 3. Alignment Scores
        print("\n[3/5] Validating alignment scores...")
        self.validate_alignment_scores()

        # 4. ChatML Format
        print("\n[4/5] Verifying ChatML format...")
        self.validate_chatml_format()

        # 5. Statistical Analysis
        print("\n[5/5] Calculating statistics...")
        self.calculate_statistics()

        # Generate report
        self.generate_report()

    def validate_json_structure(self):
        """Validate JSON structure of all files"""
        for persona_id in self.personas.keys():
            pairs_file = self.data_dir / f"{persona_id}_dpo_pairs.jsonl"
            rejected_file = self.data_dir / f"{persona_id}_dpo_pairs_rejected.jsonl"

            # Validate accepted pairs
            valid_count, invalid_count = self._validate_file(pairs_file, is_accepted=True)
            self.results['statistics'][persona_id]['valid_pairs'] = valid_count
            self.results['statistics'][persona_id]['invalid_pairs'] = invalid_count

            # Validate rejected pairs
            if rejected_file.exists():
                valid_rej, invalid_rej = self._validate_file(rejected_file, is_accepted=False)
                self.results['statistics'][persona_id]['rejected_pairs'] = valid_rej

    def _validate_file(self, file_path: Path, is_accepted: bool) -> Tuple[int, int]:
        """Validate a single JSONL file"""
        valid_count = 0
        invalid_count = 0

        if not file_path.exists():
            self.results['critical_errors'].append(f"File not found: {file_path}")
            return 0, 0

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)

                    if is_accepted:
                        # Check required fields
                        required = ['prompt', 'chosen', 'rejected', 'metadata']
                        missing = [field for field in required if field not in data]
                        if missing:
                            self.results['critical_errors'].append(
                                f"{file_path.name}:{line_num} - Missing fields: {missing}"
                            )
                            invalid_count += 1
                            continue

                        # Validate chosen/rejected are valid JSON strings
                        try:
                            json.loads(data['chosen'])
                            json.loads(data['rejected'])
                        except json.JSONDecodeError as e:
                            self.results['critical_errors'].append(
                                f"{file_path.name}:{line_num} - Invalid recipe JSON: {e}"
                            )
                            invalid_count += 1
                            continue

                    valid_count += 1

                except json.JSONDecodeError as e:
                    self.results['critical_errors'].append(
                        f"{file_path.name}:{line_num} - JSON decode error: {e}"
                    )
                    invalid_count += 1

        return valid_count, invalid_count

    def validate_persona_constraints(self):
        """Check if chosen recipes violate persona constraints"""
        for persona_id, persona in self.personas.items():
            pairs_file = self.data_dir / f"{persona_id}_dpo_pairs.jsonl"

            if not pairs_file.exists():
                continue

            forbidden = [kw.lower() for kw in persona.get('forbidden_keywords', [])]
            violations = []

            with open(pairs_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line)
                        chosen_recipe = json.loads(data['chosen'])

                        # Check recipe structure
                        if chosen_recipe.get('status') != 'ok':
                            continue

                        recipe = chosen_recipe.get('recipe', {})

                        # Check forbidden keywords in various fields
                        fields_to_check = [
                            ('name', recipe.get('name', '')),
                            ('ingredients', ' '.join(recipe.get('main_ingredients', []))),
                            ('steps', recipe.get('steps', ''))
                        ]

                        for field_name, field_value in fields_to_check:
                            field_lower = field_value.lower()
                            for forbidden_kw in forbidden:
                                if forbidden_kw in field_lower:
                                    violations.append({
                                        'line': line_num,
                                        'field': field_name,
                                        'keyword': forbidden_kw,
                                        'recipe_name': recipe.get('name', 'Unknown')
                                    })

                    except (json.JSONDecodeError, KeyError) as e:
                        # Already logged in JSON validation
                        pass

            if violations:
                self.results['violations'][persona_id] = violations
                self.results['warnings'].append(
                    f"{persona_id}: Found {len(violations)} constraint violations"
                )

    def validate_alignment_scores(self):
        """Validate alignment scores and confidence levels"""
        for persona_id in self.personas.keys():
            pairs_file = self.data_dir / f"{persona_id}_dpo_pairs.jsonl"

            if not pairs_file.exists():
                continue

            scores = {
                'chosen_higher': 0,
                'rejected_higher': 0,
                'equal': 0,
                'confidence_high': 0,
                'confidence_medium': 0,
                'confidence_low': 0,
                'score_differences': []
            }

            with open(pairs_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        eval_data = data.get('metadata', {}).get('evaluation', {})

                        # Get scores
                        variant_a = eval_data.get('variant_a_evaluation', {})
                        variant_b = eval_data.get('variant_b_evaluation', {})

                        score_a = variant_a.get('alignment_score', 0)
                        score_b = variant_b.get('alignment_score', 0)

                        chosen_variant = eval_data.get('chosen_variant', '')
                        chosen_score = score_a if chosen_variant == 'A' else score_b
                        rejected_score = score_b if chosen_variant == 'A' else score_a

                        diff = chosen_score - rejected_score
                        scores['score_differences'].append(diff)

                        if diff > 0:
                            scores['chosen_higher'] += 1
                        elif diff < 0:
                            scores['rejected_higher'] += 1
                        else:
                            scores['equal'] += 1

                        # Confidence
                        confidence = eval_data.get('confidence', 'unknown')
                        if confidence == 'high':
                            scores['confidence_high'] += 1
                        elif confidence == 'medium':
                            scores['confidence_medium'] += 1
                        elif confidence == 'low':
                            scores['confidence_low'] += 1

                    except (json.JSONDecodeError, KeyError):
                        pass

            self.results['statistics'][persona_id]['alignment_scores'] = scores

            # Warning if rejected has higher score
            if scores['rejected_higher'] > 0:
                self.results['warnings'].append(
                    f"{persona_id}: {scores['rejected_higher']} pairs have rejected > chosen score"
                )

    def validate_chatml_format(self):
        """Validate ChatML format in prompts"""
        chatml_pattern = r'<\|im_start\|>system\n.*?<\|im_end\|>\n<\|im_start\|>user\n.*?<\|im_end\|>\n<\|im_start\|>assistant\n'

        for persona_id in self.personas.keys():
            pairs_file = self.data_dir / f"{persona_id}_dpo_pairs.jsonl"

            if not pairs_file.exists():
                continue

            invalid_prompts = 0

            with open(pairs_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line)
                        prompt = data.get('prompt', '')

                        if not re.search(chatml_pattern, prompt, re.DOTALL):
                            invalid_prompts += 1
                            self.results['warnings'].append(
                                f"{persona_id}:{line_num} - Invalid ChatML format"
                            )

                    except json.JSONDecodeError:
                        pass

            self.results['statistics'][persona_id]['invalid_chatml'] = invalid_prompts

    def calculate_statistics(self):
        """Calculate overall statistics"""
        total_pairs = 0
        total_rejected = 0

        for persona_id in self.personas.keys():
            stats = self.results['statistics'][persona_id]
            valid = stats.get('valid_pairs', 0)
            rejected = stats.get('rejected_pairs', 0)

            total_pairs += valid
            total_rejected += rejected

            # Pass rate
            total_generated = valid + rejected
            pass_rate = (valid / total_generated * 100) if total_generated > 0 else 0
            stats['pass_rate'] = pass_rate

        self.results['statistics']['overall'] = {
            'total_accepted_pairs': total_pairs,
            'total_rejected_pairs': total_rejected,
            'overall_pass_rate': (total_pairs / (total_pairs + total_rejected) * 100) if (total_pairs + total_rejected) > 0 else 0
        }

    def generate_report(self):
        """Generate validation report"""
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)

        # Critical Errors
        if self.results['critical_errors']:
            print("\n🚨 CRITICAL ERRORS:")
            for error in self.results['critical_errors'][:10]:  # Show first 10
                print(f"  ❌ {error}")
            if len(self.results['critical_errors']) > 10:
                print(f"  ... and {len(self.results['critical_errors']) - 10} more")
        else:
            print("\n✅ No critical errors found")

        # Warnings
        if self.results['warnings']:
            print(f"\n⚠️  WARNINGS ({len(self.results['warnings'])}):")
            for warning in self.results['warnings'][:10]:
                print(f"  ⚠️  {warning}")
            if len(self.results['warnings']) > 10:
                print(f"  ... and {len(self.results['warnings']) - 10} more")

        # Statistics
        print("\n📊 STATISTICS:")
        print(f"\n{'Persona':<30} {'Pairs':<10} {'Pass Rate':<12} {'Confidence H/M/L'}")
        print("-" * 70)

        for persona_id in self.personas.keys():
            stats = self.results['statistics'].get(persona_id, {})
            pairs = stats.get('valid_pairs', 0)
            pass_rate = stats.get('pass_rate', 0)

            scores = stats.get('alignment_scores', {})
            conf_high = scores.get('confidence_high', 0)
            conf_med = scores.get('confidence_medium', 0)
            conf_low = scores.get('confidence_low', 0)

            print(f"{persona_id:<30} {pairs:<10} {pass_rate:>6.1f}%      {conf_high}/{conf_med}/{conf_low}")

        # Overall
        overall = self.results['statistics'].get('overall', {})
        total = overall.get('total_accepted_pairs', 0)
        overall_pass = overall.get('overall_pass_rate', 0)

        print("-" * 70)
        print(f"{'TOTAL':<30} {total:<10} {overall_pass:>6.1f}%")

        # Violations
        if self.results['violations']:
            print("\n🚫 CONSTRAINT VIOLATIONS:")
            for persona_id, violations in self.results['violations'].items():
                print(f"\n  {persona_id}: {len(violations)} violations")
                for v in violations[:3]:
                    print(f"    - Line {v['line']}: '{v['keyword']}' in {v['field']} ({v['recipe_name']})")
                if len(violations) > 3:
                    print(f"    ... and {len(violations) - 3} more")

        # Final recommendation
        print("\n" + "=" * 60)
        if not self.results['critical_errors']:
            print("✅ RECOMMENDATION: Data is ready for DPO training")
            print(f"   Total pairs: {total}")
            print(f"   Pass rate: {overall_pass:.1f}%")
        else:
            print("❌ RECOMMENDATION: Fix critical errors before training")
            print(f"   Critical errors: {len(self.results['critical_errors'])}")
        print("=" * 60)

        # Save detailed report
        report_file = self.data_dir / "validation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\n📄 Detailed report saved to: {report_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate DPO training data")
    parser.add_argument("--data_dir", default="../../data/dpo_final_pairs",
                       help="Directory with DPO pairs")
    parser.add_argument("--personas_config", default="personas.yaml",
                       help="Personas configuration file")
    args = parser.parse_args()

    validator = DPODataValidator(args.data_dir, args.personas_config)
    validator.validate_all()


if __name__ == "__main__":
    main()
