#!/usr/bin/env python3
"""
Agent Lightning CLI - Fine-tuning and behavior optimization for MCP agents.

This script provides commands to:
- Build datasets from rewarded episodes
- Run fine-tuning jobs
- Promote and rollback tuned models
- View training history and lineage

Usage:
    python -m lightning.cli build-dataset --agent-id mcp-agents --name v1 --min-reward 0.5
    python -m lightning.cli train --dataset-id <id> --agent-id mcp-agents
    python -m lightning.cli promote --run-id <id> --agent-id mcp-agents
    python -m lightning.cli rollback --agent-id mcp-agents --reason "Performance regression"
    python -m lightning.cli status --agent-id mcp-agents
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightning import (
    RLLedgerCosmos,
    DatasetBuilder,
    TrainingRunner,
    DeploymentRegistry,
    RewardWriter,
    RewardSource,
    TrainingStatus,
    get_rl_ledger,
    get_dataset_builder,
    get_training_runner,
    get_deployment_registry,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_build_dataset(args):
    """Build a fine-tuning dataset from rewarded episodes."""
    builder = get_dataset_builder()
    
    sources = None
    if args.sources:
        sources = [RewardSource(s) for s in args.sources.split(',')]
    
    dataset = builder.build_dataset(
        agent_id=args.agent_id,
        name=args.name,
        description=args.description,
        min_reward=args.min_reward,
        sources=sources,
    )
    
    if dataset:
        print(f"\n✅ Dataset created successfully!")
        print(f"   ID: {dataset.id}")
        print(f"   Name: {dataset.name}")
        print(f"   Training examples: {dataset.training_count}")
        print(f"   Validation examples: {dataset.validation_count}")
        print(f"   Training file: {dataset.local_path}")
        print(f"   Validation file: {dataset.metadata.get('validation_path')}")
        return 0
    else:
        print("\n❌ Failed to create dataset")
        return 1


def cmd_build_golden(args):
    """Build a dataset from golden conversations."""
    builder = get_dataset_builder()
    
    dataset = builder.build_from_golden_conversations(
        agent_id=args.agent_id,
        name=args.name,
        golden_file=args.golden_file,
        description=args.description,
    )
    
    if dataset:
        print(f"\n✅ Golden dataset created successfully!")
        print(f"   ID: {dataset.id}")
        print(f"   Name: {dataset.name}")
        print(f"   Training examples: {dataset.training_count}")
        print(f"   Validation examples: {dataset.validation_count}")
        return 0
    else:
        print("\n❌ Failed to create golden dataset")
        return 1


def cmd_list_datasets(args):
    """List all datasets for an agent."""
    builder = get_dataset_builder()
    datasets = builder.list_datasets(args.agent_id)
    
    if not datasets:
        print(f"\nNo datasets found for agent '{args.agent_id}'")
        return 0
    
    print(f"\n📦 Datasets for agent '{args.agent_id}':")
    print("-" * 80)
    
    for ds in datasets:
        print(f"  ID: {ds.id}")
        print(f"  Name: {ds.name}")
        print(f"  Created: {ds.created_at}")
        print(f"  Training: {ds.training_count} | Validation: {ds.validation_count}")
        print(f"  Reward threshold: {ds.reward_threshold}")
        print("-" * 80)
    
    return 0


def cmd_train(args):
    """Start a fine-tuning job."""
    runner = get_training_runner()
    
    hyperparams = None
    if args.epochs:
        hyperparams = {"n_epochs": args.epochs}
    
    print(f"\n🚀 Starting fine-tuning job...")
    print(f"   Dataset ID: {args.dataset_id}")
    print(f"   Base model: {args.base_model or 'default'}")
    
    run = runner.run_training(
        dataset_id=args.dataset_id,
        agent_id=args.agent_id,
        base_model=args.base_model,
        hyperparameters=hyperparams,
        wait=not args.no_wait,
    )
    
    if not run:
        print("\n❌ Failed to start training")
        return 1
    
    print(f"\n   Run ID: {run.id}")
    print(f"   Status: {run.status.value}")
    
    if run.status == TrainingStatus.SUCCEEDED:
        print(f"\n✅ Training completed successfully!")
        print(f"   Tuned model: {run.tuned_model_name}")
        print(f"\n   To promote this model, run:")
        print(f"   python -m lightning.cli promote --run-id {run.id} --agent-id {args.agent_id}")
        return 0
    elif run.status == TrainingStatus.FAILED:
        print(f"\n❌ Training failed: {run.error_message}")
        return 1
    else:
        print(f"\n⏳ Training is in progress (run ID: {run.id})")
        print(f"   Check status with: python -m lightning.cli status --run-id {run.id} --agent-id {args.agent_id}")
        return 0


def cmd_list_runs(args):
    """List training runs for an agent."""
    runner = get_training_runner()
    
    status = None
    if args.status:
        status = TrainingStatus(args.status)
    
    runs = runner.list_runs(args.agent_id, status)
    
    if not runs:
        print(f"\nNo training runs found for agent '{args.agent_id}'")
        return 0
    
    print(f"\n🏋️ Training runs for agent '{args.agent_id}':")
    print("-" * 80)
    
    for run in runs:
        status_emoji = {
            "pending": "⏳",
            "running": "🔄",
            "succeeded": "✅",
            "failed": "❌",
            "cancelled": "⛔",
        }.get(run.status.value, "❓")
        
        print(f"  {status_emoji} {run.id}")
        print(f"     Status: {run.status.value}")
        print(f"     Base model: {run.base_model}")
        print(f"     Dataset: {run.dataset_id}")
        if run.tuned_model_name:
            print(f"     Tuned model: {run.tuned_model_name}")
        print(f"     Created: {run.created_at}")
        print("-" * 80)
    
    return 0


def cmd_check_status(args):
    """Check status of a training run."""
    runner = get_training_runner()
    
    run = runner.check_status(args.run_id, args.agent_id)
    
    if not run:
        print(f"\n❌ Training run '{args.run_id}' not found")
        return 1
    
    print(f"\n📊 Training run status:")
    print(f"   ID: {run.id}")
    print(f"   Status: {run.status.value}")
    print(f"   Base model: {run.base_model}")
    print(f"   Dataset: {run.dataset_id}")
    
    if run.started_at:
        print(f"   Started: {run.started_at}")
    if run.completed_at:
        print(f"   Completed: {run.completed_at}")
    if run.tuned_model_name:
        print(f"   Tuned model: {run.tuned_model_name}")
    if run.error_message:
        print(f"   Error: {run.error_message}")
    
    return 0


def cmd_promote(args):
    """Promote a tuned model to active deployment."""
    registry = get_deployment_registry()
    
    deployment = registry.promote(
        agent_id=args.agent_id,
        training_run_id=args.run_id,
        promoted_by=args.promoted_by,
    )
    
    if deployment:
        print(f"\n✅ Model promoted successfully!")
        print(f"   Deployment ID: {deployment.id}")
        print(f"   Tuned model: {deployment.tuned_model_name}")
        print(f"   Agent ID: {deployment.agent_id}")
        print(f"\n   The agent will now use this tuned model when USE_TUNED_MODEL=true")
        return 0
    else:
        print("\n❌ Failed to promote model")
        return 1


def cmd_rollback(args):
    """Rollback to a previous deployment."""
    registry = get_deployment_registry()
    
    deployment = registry.rollback(
        agent_id=args.agent_id,
        target_deployment_id=args.target_id,
        reason=args.reason,
        rolled_back_by=args.rolled_back_by,
    )
    
    if deployment:
        print(f"\n✅ Rolled back successfully!")
        print(f"   Deployment ID: {deployment.id}")
        print(f"   Tuned model: {deployment.tuned_model_name}")
        print(f"   Rollback reason: {args.reason}")
        return 0
    else:
        print("\n❌ Failed to rollback")
        return 1


def cmd_deactivate(args):
    """Deactivate the current deployment (revert to base model)."""
    registry = get_deployment_registry()
    
    success = registry.deactivate(args.agent_id, reason=args.reason)
    
    if success:
        print(f"\n✅ Deployment deactivated for agent '{args.agent_id}'")
        print(f"   The agent will now use the base model")
        return 0
    else:
        print("\n❌ Failed to deactivate deployment")
        return 1


def cmd_list_deployments(args):
    """List deployment history for an agent."""
    registry = get_deployment_registry()
    deployments = registry.list_deployments(args.agent_id)
    
    if not deployments:
        print(f"\nNo deployments found for agent '{args.agent_id}'")
        return 0
    
    print(f"\n🚀 Deployments for agent '{args.agent_id}':")
    print("-" * 80)
    
    for dep in deployments:
        active_badge = "🟢 ACTIVE" if dep.is_active else "⚪"
        print(f"  {active_badge} {dep.id}")
        print(f"     Model: {dep.tuned_model_name}")
        print(f"     Promoted: {dep.promoted_at}")
        if dep.promoted_by:
            print(f"     Promoted by: {dep.promoted_by}")
        if dep.rollback_from:
            print(f"     Rollback from: {dep.rollback_from}")
            print(f"     Rollback reason: {dep.rollback_reason}")
        print("-" * 80)
    
    return 0


def cmd_lineage(args):
    """Show deployment lineage (full history with training/dataset info)."""
    registry = get_deployment_registry()
    lineage = registry.get_deployment_lineage(args.agent_id)
    
    if not lineage:
        print(f"\nNo lineage found for agent '{args.agent_id}'")
        return 0
    
    print(f"\n📜 Deployment lineage for agent '{args.agent_id}':")
    print("=" * 80)
    
    for entry in lineage:
        active = "🟢 ACTIVE" if entry.get("is_active") else ""
        print(f"\n{active}")
        print(f"Deployment: {entry['deployment_id']}")
        print(f"  Model: {entry['tuned_model']}")
        print(f"  Promoted: {entry['promoted_at']}")
        
        if entry.get("training_run"):
            tr = entry["training_run"]
            print(f"  Training Run: {tr['id']}")
            print(f"    Base model: {tr['base_model']}")
            print(f"    Status: {tr['status']}")
        
        if entry.get("dataset"):
            ds = entry["dataset"]
            print(f"  Dataset: {ds['id']} ({ds['name']})")
            print(f"    Examples: {ds['training_count']} train, {ds['validation_count']} val")
            print(f"    Reward threshold: {ds['reward_threshold']}")
        
        print("-" * 80)
    
    return 0


def cmd_health(args):
    """Check health of the RL Ledger."""
    ledger = get_rl_ledger()
    health = ledger.health_check()
    
    print("\n🏥 RL Ledger Health Check:")
    print(f"   Initialized: {health['initialized']}")
    print(f"   Endpoint: {health['endpoint']}")
    print(f"   Database: {health['database']}")
    print(f"   Auth mode: {health['auth_mode']}")
    
    if health.get('containers'):
        print("\n   Containers:")
        for name, status in health['containers'].items():
            emoji = "✅" if status == "healthy" else "❌"
            print(f"     {emoji} {name}: {status}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Agent Lightning CLI - Fine-tuning and behavior optimization"
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # build-dataset command
    build_ds = subparsers.add_parser('build-dataset', help='Build dataset from rewarded episodes')
    build_ds.add_argument('--agent-id', required=True, help='Agent ID')
    build_ds.add_argument('--name', required=True, help='Dataset name')
    build_ds.add_argument('--description', help='Dataset description')
    build_ds.add_argument('--min-reward', type=float, default=0.0, help='Minimum avg reward for inclusion')
    build_ds.add_argument('--sources', help='Comma-separated reward sources to filter by')
    build_ds.set_defaults(func=cmd_build_dataset)
    
    # build-golden command
    build_golden = subparsers.add_parser('build-golden', help='Build dataset from golden conversations')
    build_golden.add_argument('--agent-id', required=True, help='Agent ID')
    build_golden.add_argument('--name', required=True, help='Dataset name')
    build_golden.add_argument('--golden-file', required=True, help='Path to golden conversations JSONL file')
    build_golden.add_argument('--description', help='Dataset description')
    build_golden.set_defaults(func=cmd_build_golden)
    
    # list-datasets command
    list_ds = subparsers.add_parser('list-datasets', help='List datasets')
    list_ds.add_argument('--agent-id', required=True, help='Agent ID')
    list_ds.set_defaults(func=cmd_list_datasets)
    
    # train command
    train = subparsers.add_parser('train', help='Start fine-tuning job')
    train.add_argument('--dataset-id', required=True, help='Dataset ID to use')
    train.add_argument('--agent-id', required=True, help='Agent ID')
    train.add_argument('--base-model', help='Base model to fine-tune')
    train.add_argument('--epochs', type=int, help='Number of epochs')
    train.add_argument('--no-wait', action='store_true', help='Do not wait for completion')
    train.set_defaults(func=cmd_train)
    
    # list-runs command
    list_runs = subparsers.add_parser('list-runs', help='List training runs')
    list_runs.add_argument('--agent-id', required=True, help='Agent ID')
    list_runs.add_argument('--status', help='Filter by status')
    list_runs.set_defaults(func=cmd_list_runs)
    
    # status command
    status = subparsers.add_parser('status', help='Check training run status')
    status.add_argument('--run-id', required=True, help='Training run ID')
    status.add_argument('--agent-id', required=True, help='Agent ID')
    status.set_defaults(func=cmd_check_status)
    
    # promote command
    promote = subparsers.add_parser('promote', help='Promote tuned model to active')
    promote.add_argument('--run-id', required=True, help='Training run ID')
    promote.add_argument('--agent-id', required=True, help='Agent ID')
    promote.add_argument('--promoted-by', help='Who is promoting')
    promote.set_defaults(func=cmd_promote)
    
    # rollback command
    rollback = subparsers.add_parser('rollback', help='Rollback to previous deployment')
    rollback.add_argument('--agent-id', required=True, help='Agent ID')
    rollback.add_argument('--target-id', help='Target deployment ID (optional, defaults to previous)')
    rollback.add_argument('--reason', help='Reason for rollback')
    rollback.add_argument('--rolled-back-by', help='Who is rolling back')
    rollback.set_defaults(func=cmd_rollback)
    
    # deactivate command
    deactivate = subparsers.add_parser('deactivate', help='Deactivate current deployment')
    deactivate.add_argument('--agent-id', required=True, help='Agent ID')
    deactivate.add_argument('--reason', help='Reason for deactivation')
    deactivate.set_defaults(func=cmd_deactivate)
    
    # list-deployments command
    list_deps = subparsers.add_parser('list-deployments', help='List deployment history')
    list_deps.add_argument('--agent-id', required=True, help='Agent ID')
    list_deps.set_defaults(func=cmd_list_deployments)
    
    # lineage command
    lineage = subparsers.add_parser('lineage', help='Show deployment lineage')
    lineage.add_argument('--agent-id', required=True, help='Agent ID')
    lineage.set_defaults(func=cmd_lineage)
    
    # health command
    health = subparsers.add_parser('health', help='Check RL Ledger health')
    health.set_defaults(func=cmd_health)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
