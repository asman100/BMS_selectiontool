# bms_selector_final.py

import pandas as pd
import pulp

def find_optimal_controllers(panel_requirements, controllers_df):
    """
    Solves for the most cost-effective combination of controllers,
    intelligently allocating universal I/O points (UIO).
    """
    prob = pulp.LpProblem(f"BMS_Controller_Selection_{panel_requirements.name}", pulp.LpMinimize)

    # --- DECISION VARIABLES ---
    controller_qty_vars = pulp.LpVariable.dicts("Qty", controllers_df['Name'], lowBound=0, cat='Integer')
    uio_as_input_vars = pulp.LpVariable.dicts("UIO_as_Input", controllers_df['Name'], lowBound=0, cat='Continuous')
    uio_as_output_vars = pulp.LpVariable.dicts("UIO_as_Output", controllers_df['Name'], lowBound=0, cat='Continuous')

    # --- OBJECTIVE FUNCTION (Minimize Cost) ---
    prob += pulp.lpSum(
        [controllers_df.loc[i, 'Cost'] * controller_qty_vars[controllers_df.loc[i, 'Name']] for i in controllers_df.index]
    ), "Total_Cost"

    # --- CONSTRAINTS ---
    for i in controllers_df.index:
        model_name = controllers_df.loc[i, 'Name']
        available_uio = controllers_df.loc[i, 'UIO'] * controller_qty_vars[model_name]
        prob += uio_as_input_vars[model_name] + uio_as_output_vars[model_name] <= available_uio, f"UIO_Allocation_{model_name}"

    total_required_inputs = panel_requirements['DI'] + panel_requirements['AI']
    total_required_outputs = panel_requirements['DO'] + panel_requirements['AO']

    total_provided_inputs = pulp.lpSum(
        [(controllers_df.loc[i, 'DI'] + controllers_df.loc[i, 'AI'] + controllers_df.loc[i, 'UI']) * controller_qty_vars[controllers_df.loc[i, 'Name']] for i in controllers_df.index]
    ) + pulp.lpSum(uio_as_input_vars)
    
    total_provided_outputs = pulp.lpSum(
        [(controllers_df.loc[i, 'DO'] + controllers_df.loc[i, 'AO'] + controllers_df.loc[i, 'UO']) * controller_qty_vars[controllers_df.loc[i, 'Name']] for i in controllers_df.index]
    ) + pulp.lpSum(uio_as_output_vars)

    prob += total_provided_inputs >= total_required_inputs, "Total_Input_Requirement"
    prob += total_provided_outputs >= total_required_outputs, "Total_Output_Requirement"

    # --- SOLVE ---
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # --- EXTRACT RESULTS ---
    if pulp.LpStatus[prob.status] == 'Optimal':
        total_cost = pulp.value(prob.objective)
        solution = {}
        for v in prob.variables():
            if v.varValue > 0 and v.name.startswith("Qty"):
                controller_name = v.name.replace("Qty_", "").replace("_", " ")
                solution[controller_name] = int(v.varValue)
        return total_cost, solution
    else:
        return None, None

# --- Main Script Execution ---
if __name__ == "__main__":
    try:
        controllers = pd.read_csv('controllers.csv').fillna(0)
        panels = pd.read_csv('panels.csv')
        
        print("✅ Successfully loaded controllers.csv and panels.csv\n")
        
        panels.set_index('PanelName', inplace=True)
        
        # This list will store all the results before saving to CSV
        all_solutions = []

        for panel_name, requirements in panels.iterrows():
            print("-" * 40)
            print(f"🔍 Solving for Panel: {panel_name}")
            print(f"   Requirements -> DI: {requirements['DI']}, DO: {requirements['DO']}, AI: {requirements['AI']}, AO: {requirements['AO']}")
            
            cost, result = find_optimal_controllers(requirements, controllers)

            if result:
                print(f"✅ Optimal Solution Found!")
                print(f"   Total Cost: ${cost:.2f}")
                print("   Controller Bill of Materials:")
                for controller, qty in result.items():
                    print(f"     - {qty} x {controller}")
                    # Add each controller row to our results list
                    all_solutions.append({
                        'PanelName': panel_name,
                        'ControllerName': controller,
                        'Quantity': qty,
                        'TotalPanelCost': cost
                    })
                print("-" * 40 + "\n")
            else:
                print(f"❌ No optimal solution could be found for {panel_name}.")
                all_solutions.append({
                    'PanelName': panel_name,
                    'ControllerName': 'No Solution Found',
                    'Quantity': 0,
                    'TotalPanelCost': 0
                })
                print("-" * 40 + "\n")
        
        # After the loop, convert the list of solutions to a DataFrame
        if all_solutions:
            solution_df = pd.DataFrame(all_solutions)
            # Save the DataFrame to a CSV file
            output_filename = 'panel_solutions.csv'
            solution_df.to_csv(output_filename, index=False)
            print(f"🎉 All panels processed. Results saved to '{output_filename}'.")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure 'controllers.csv' and 'panels.csv' are in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")