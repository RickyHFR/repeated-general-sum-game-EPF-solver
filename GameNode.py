from typing import Union

from matplotlib import pyplot as plt
import numpy as np
from EPF import EPF
from matplotlib.widgets import Button, Slider

class GameNode:
    def __init__(self,
                 player: int,
                 payoff: Union[tuple[float, float], EPF] = None,
                 discount_factor: float=0.9,
                 max_depth: int=50):
        self.player = player       
        self.payoff = payoff       
        self.parent = None
        self.children = []        
        self.discount_factor = discount_factor
        self.max_depth = max_depth
        self.root = None
        self.grim_memoization = {}
        self.EPF_memoization = {}

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.set_parent(self)

    def set_parent(self, parent_node):
        self.parent = parent_node

    def set_root(self, nodes):
        for node in nodes:
            node.root = self

    def get_grim_value(self, curr_depth = 0):
        if self.root.grim_memoization.get((self, curr_depth)) is not None:
            return self.root.grim_memoization[(self, curr_depth)]
        else:
            if self.player == -1 and curr_depth == self.root.max_depth:
                self.root.grim_memoization[(self, curr_depth)] = self.payoff[0] * self.discount_factor ** curr_depth
            elif self.player == -1 and curr_depth < self.root.max_depth:
                self.root.grim_memoization[(self, curr_depth)] = self.payoff[0] * self.discount_factor ** curr_depth + self.root.get_grim_value(curr_depth + 1) * self.discount_factor ** (curr_depth + 1)
            elif self.player == 0:
                self.root.grim_memoization[(self, curr_depth)] = min([child.get_grim_value(curr_depth) for child in self.children])
            else:
                self.root.grim_memoization[(self, curr_depth)] = max([child.get_grim_value(curr_depth) for child in self.children])
        return self.root.grim_memoization[(self, curr_depth)] 

    def get_EPF(self, curr_depth = 0):
        if self.root.EPF_memoization.get((self, curr_depth)) is not None:
            return self.root.EPF_memoization[(self, curr_depth)]
        else:
            if self.player == -1 and curr_depth == self.root.max_depth:
                self.root.EPF_memoization[self, curr_depth] = EPF(np.array([[self.payoff[0], self.payoff[1]]])).scale_then_shift((0, 0), self.discount_factor ** curr_depth)
            elif self.player == -1 and curr_depth < self.root.max_depth:
                self.root.EPF_memoization[self, curr_depth] = self.root.get_EPF(curr_depth + 1).scale_then_shift(
                    (self.payoff[0] * self.discount_factor ** curr_depth, self.payoff[1] * self.discount_factor ** curr_depth),
                    self.discount_factor)
            if self.player == 0:
                result_EPF = EPF(None)
                for child in self.children:
                    result_EPF = result_EPF.find_concave_envelope(child.get_EPF(curr_depth))
                self.root.EPF_memoization[self, curr_depth] = result_EPF
            if self.player == 1:
                result_EPF = EPF(None)
                for child in self.children:
                    child_threshold = max([other_child.get_grim_value(curr_depth) for other_child in self.children if other_child != child])
                    result_EPF = result_EPF.find_concave_envelope(child.get_EPF(curr_depth))
                    result_EPF = result_EPF.left_truncate(child_threshold)
                self.root.EPF_memoization[self, curr_depth] = result_EPF
            return self.root.EPF_memoization[self, curr_depth]
        
    def solve_EPF(self):
        if self.root.EPF_memoization.get((self, 0)) is None:
            self.get_EPF()
        return self.root.EPF_memoization[(self, 0)]
    
    def solve_grim(self):
        if self.root.grim_memoization.get((self, 0)) is None:
            self.get_grim_value()
        return self.root.grim_memoization[(self, 0)]
    
    def draw_EPF(self):
        # Ensure all EPF values are precomputed
        if self.root.EPF_memoization.get((self, 0)) is None:
            self.get_EPF()  # Precompute all required EPF values

        dict_EPF = self.root.EPF_memoization

        # Create a figure with two subplots: main plot (left) and table (right)
        fig, (ax_main, ax_table) = plt.subplots(ncols=2,
                                                gridspec_kw={'width_ratios': [3, 1]},
                                                figsize=(14, 8))
        plt.subplots_adjust(left=0.1, bottom=0.25)

        # Create slider axes (placed at the bottom across the full figure width)
        depth_slider_ax = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        depth_slider = Slider(depth_slider_ax, 'Depth', 0, self.root.max_depth, valinit=0, valstep=1)

        # Create button axes for toggling axis mode (position can be adjusted)
        button_ax = plt.axes([0.1, 0.02, 0.15, 0.04])
        toggle_button = Button(button_ax, 'Fixed Axis')  # Initial mode: fixed axis

        # Compute global axis limits for the main plot (fixed axis)
        all_knots = np.vstack([epf.knots for epf in dict_EPF.values()])
        global_x_min, global_x_max = np.min(all_knots[:, 0]), np.max(all_knots[:, 0])
        global_y_min, global_y_max = np.min(all_knots[:, 1]), np.max(all_knots[:, 1])
        ax_main.set_xlim(global_x_min, global_x_max)
        ax_main.set_ylim(global_y_min, global_y_max)

        # Set axis labels and an initial title for the main plot
        ax_main.set_xlabel("Follower")
        ax_main.set_ylabel("Leader")
        ax_main.set_title("EPF at depth 0")

        # Turn off the table axes (frame, ticks, etc.)
        ax_table.axis('off')

        # Initialize main plot with depth 0
        initial_EPF = dict_EPF[(self, 0)]
        line, = ax_main.plot(initial_EPF.knots[:, 0],
                            initial_EPF.knots[:, 1],
                            marker='o')

        # Initialize text labels for the main plot (to annotate each point)
        text_labels = []
        
        # Variable to hold the table object (for removal/updating)
        table_obj = None

        # Mode variable: True means fixed axis, False means dynamic axis
        fixed_axis_mode = True

        def update(val):
            nonlocal table_obj, fixed_axis_mode
            depth = int(depth_slider.val)
            EPF_i = dict_EPF[(self, depth)]
            
            # Update the main plot line data
            line.set_xdata(EPF_i.knots[:, 0])
            line.set_ydata(EPF_i.knots[:, 1])
            
            # Remove old text labels
            while text_labels:
                label = text_labels.pop()
                label.remove()
                
            # Add new text labels at each point on the main plot
            for x, y in EPF_i.knots:
                text_labels.append(ax_main.text(x, y, f'({x:.2f}, {y:.2f})',
                                                fontsize=9, ha='right'))
            
            # Update the title with current depth
            ax_main.set_title(f"EPF at depth {depth}")

            # Update the axis limits based on the selected mode
            if fixed_axis_mode:
                ax_main.set_xlim(global_x_min, global_x_max)
                ax_main.set_ylim(global_y_min, global_y_max)
            else:
                local_x_min, local_x_max = np.min(EPF_i.knots[:, 0]), np.max(EPF_i.knots[:, 0])
                local_y_min, local_y_max = np.min(EPF_i.knots[:, 1]), np.max(EPF_i.knots[:, 1])
                ax_main.set_xlim(local_x_min, local_x_max)
                ax_main.set_ylim(local_y_min, local_y_max)

            # Update the table on the right-hand side with increased precision (4 decimals)
            if table_obj is not None:
                table_obj.remove()  # Remove the previous table
            data = [[f"{x:.4f}", f"{y:.4f}"] for x, y in EPF_i.knots]
            table_obj = ax_table.table(cellText=data,
                                    colLabels=["Follower", "Leader"],
                                    loc='center')
            # Optional table formatting
            table_obj.auto_set_font_size(False)
            table_obj.set_fontsize(10)
            table_obj.scale(1, 1.5)

            fig.canvas.draw_idle()  # Refresh the figure

        def toggle_axis_mode(event):
            nonlocal fixed_axis_mode
            fixed_axis_mode = not fixed_axis_mode
            # Update the button label to reflect the current mode
            toggle_button.label.set_text("Fixed Axis" if fixed_axis_mode else "Dynamic Axis")
            # Force an update so the axis limits change immediately
            update(None)

        # Connect the slider and button callbacks
        depth_slider.on_changed(update)
        toggle_button.on_clicked(toggle_axis_mode)

        plt.show()



