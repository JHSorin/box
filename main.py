import time
import sys
import os
import yaml

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from perception.sensor import Sensor
from perception.processor import Processor
from planning.pattern import Pattern
from planning.solver import Solver
from execution.arm_interface import ArmInterface
from execution.trajectory import TrajectoryPlanner
from visualization.viewer_2d import Viewer2D
# from visualization.viewer_3d import Viewer3D # Deprecated
from visualization.dashboard import Dashboard
from utils.simulation import VirtualPallet

def main():
    print("Initializing Palletizing System (Algorithm Validation Mode)...")
    
    # Load Box Config for simulation
    with open('config/box_config.yaml', 'r') as f:
        box_config = yaml.safe_load(f)
        box_dims = box_config['dimensions'] # L, W, H
        
    # 1. Initialize Virtual World
    virtual_pallet = VirtualPallet()
    
    # 2. Initialize Modules
    sensor = Sensor(mode='simulation', virtual_source=virtual_pallet)
    processor = Processor()
    pattern_gen = Pattern() 
    solver = Solver() 
    arm = ArmInterface()
    planner = TrajectoryPlanner()
    viewer = Viewer2D(box_dims=box_dims)
    dashboard = Dashboard()
    
    print("System Started.")
    
    current_layer = 0
    
    try:
        print(f"\n=== Processing Single Layer ===")
        
        # Loop until this layer is full
        layer_full = False
        while not layer_full:
            
            # 2. Perception
            rgb, depth = sensor.get_rgbd()
            height_map = processor.build_height_map(depth)
            
            # 3. Planning
            # Get template for this layer
            slots = pattern_gen.get_pattern()
            
            # Solver finds the NEXT valid slot
            # For single layer, we check near z=0 (or whatever base height is)
            layer_z = 0 
            
            # Note: Solver now iterates through slots and returns the first one 
            # that is deemed "empty" in the height map.
            target_pose = solver.find_slot(height_map, slots, layer_z)
            
            if target_pose:
                print(f"Plan: Found target at {target_pose}")
                
                # 4. Visualization
                # Use OpenCV Viewer
                viewer.update(virtual_pallet.get_rgb_image(), target_pose=target_pose)
                
                dashboard.update(rgb, f"Layer {current_layer}: Placing Box")
                
                # Interactive Step: Wait for user to inspect the plan
                viewer.wait_for_input()
                
                # 5. Execution
                waypoints = planner.generate_place_trajectory(target_pose)
                # Simulate execution (skipping print noise)
                time.sleep(0.1) 
                
                # 6. Update World
                # Important: Solver depends on the world being updated!
                virtual_pallet.add_box(
                    target_pose['x'], 
                    target_pose['y'], 
                    box_dims[0], 
                    box_dims[1], 
                    box_dims[2], 
                    target_pose['theta'],
                    is_obstacle=target_pose.get('forced', False),
                    layer_index=current_layer
                )
                print(f"World: Box placed at ({target_pose['x']}, {target_pose['y']})")
                
            else:
                # No more valid slots found -> Layer Full
                print(f"Plan: No valid slots left. Packing Complete.")
                layer_full = True
                time.sleep(1.0)
                    
    except KeyboardInterrupt:
        print("System stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        viewer.close()
    
    print("\nSystem Shutdown.")

if __name__ == "__main__":
    main()
