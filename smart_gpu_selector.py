#!/usr/bin/env python3

import subprocess
import os
import sys
import time

def get_gpu_utilization():
    """Get GPU utilization and memory usage"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,utilization.gpu,memory.used,memory.total', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [x.strip() for x in line.split(',')]
                gpu_id = int(parts[0])
                utilization = int(parts[1])
                mem_used = int(parts[2])
                mem_total = int(parts[3])
                mem_percent = (mem_used / mem_total) * 100
                
                gpus.append({
                    'id': gpu_id,
                    'utilization': utilization,
                    'memory_used': mem_used,
                    'memory_total': mem_total,
                    'memory_percent': mem_percent
                })
        return gpus
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []

def find_best_gpu_pair(gpus, threshold_util=20, threshold_mem=20):
    """Find the best pair of GPUs for training"""
    
    print("🔍 GPU Status Analysis:")
    print("=" * 60)
    
    available_gpus = []
    for gpu in gpus:
        status = "🟢 AVAILABLE" if (gpu['utilization'] < threshold_util and 
                                   gpu['memory_percent'] < threshold_mem) else "🔴 BUSY"
        
        print(f"GPU {gpu['id']}: {status}")
        print(f"  Utilization: {gpu['utilization']}%")
        print(f"  Memory: {gpu['memory_used']}/{gpu['memory_total']}MB ({gpu['memory_percent']:.1f}%)")
        print()
        
        if gpu['utilization'] < threshold_util and gpu['memory_percent'] < threshold_mem:
            available_gpus.append(gpu['id'])
    
    if len(available_gpus) >= 2:
        # Select first two available GPUs
        selected = available_gpus[:2]
        print(f"✅ Selected GPUs for training: {selected}")
        return selected
    elif len(available_gpus) == 1:
        print(f"⚠️  Only 1 GPU available: {available_gpus[0]}")
        print("   Consider single GPU training or wait for more GPUs")
        return available_gpus
    else:
        print("❌ No GPUs available for training")
        return []

def launch_training(gpu_list):
    """Launch training with selected GPUs"""
    if len(gpu_list) < 2:
        print("❌ Need at least 2 GPUs for dual GPU training")
        print("💡 Options:")
        print("   1. Wait for GPUs to become available")
        print("   2. Use single GPU training")
        print("   3. Force use occupied GPUs (not recommended)")
        return False
    
    # Set environment variable
    gpu_str = ','.join(map(str, gpu_list))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    
    print(f"\n🚀 Launching training on GPUs {gpu_list}")
    print(f"   CUDA_VISIBLE_DEVICES={gpu_str}")
    print("=" * 60)
    
    # Launch training command
    cmd = [
        'torchrun',
        '--nproc_per_node=2',
        '--master_port=29500',
        'train.py'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return False

def main():
    print("🤖 Smart GPU Selector for CodeT5p Training")
    print("=" * 60)
    
    # Get GPU information
    gpus = get_gpu_utilization()
    if not gpus:
        print("❌ Could not detect GPUs")
        sys.exit(1)
    
    # Find best GPU pair
    selected_gpus = find_best_gpu_pair(gpus)
    
    if len(selected_gpus) >= 2:
        # Ask for confirmation
        print(f"\n❓ Proceed with training on GPUs {selected_gpus}? (y/n): ", end="")
        response = input().lower().strip()
        
        if response in ['y', 'yes']:
            success = launch_training(selected_gpus)
            sys.exit(0 if success else 1)
        else:
            print("🛑 Training cancelled by user")
            sys.exit(0)
    else:
        print("\n💡 Manual GPU selection options:")
        print("   export CUDA_VISIBLE_DEVICES=0,2  # Use GPU 0 and 2")
        print("   export CUDA_VISIBLE_DEVICES=1,2  # Use GPU 1 and 2")
        print("   export CUDA_VISIBLE_DEVICES=0,1  # Use GPU 0 and 1")
        print("   ./launch_dual_gpu.sh")
        sys.exit(1)

if __name__ == "__main__":
    main()