# bms_selector_with_boq.py

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
        solution = {}
        for v in prob.variables():
            if v.varValue > 0 and v.name.startswith("Qty"):
                controller_name = v.name.replace("Qty_", "").replace("_", " ")
                solution[controller_name] = int(v.varValue)
        return solution
    else:
        return None

# --- Main Script Execution ---
if __name__ == "__main__":
    try:
        controllers = pd.read_csv('controllers.csv').fillna(0)
        panels = pd.read_csv('panels.csv')
        
        print("✅ Successfully loaded controllers.csv and panels.csv\n")
        
        panels.set_index('PanelName', inplace=True)
        
        all_solutions = []

        for panel_name, requirements in panels.iterrows():
            print(f"🔍 Solving for Panel: {panel_name}")
            result = find_optimal_controllers(requirements, controllers)

            if result:
                print(f"✅ Optimal Solution Found!")
                for controller, qty in result.items():
                    all_solutions.append({
                        'PanelName': panel_name,
                        'ControllerName': controller,
                        'Quantity': qty
                    })
            else:
                print(f"❌ No optimal solution could be found for {panel_name}.")
                all_solutions.append({
                    'PanelName': panel_name,
                    'ControllerName': 'No Solution Found',
                    'Quantity': 0
                })
        
        if not all_solutions:
            print("No solutions were found for any panels.")
        else:
            # --- GENERATE AND SAVE OUTPUTS ---
            solution_df = pd.DataFrame(all_solutions)

            # 1. Create the Pivoted Panel Matrix Output
            print("\nGenerating Pivoted Panel Matrix...")
            pivoted_df = solution_df.pivot_table(
                index='PanelName',
                columns='ControllerName',
                values='Quantity',
                fill_value=0
            )
            pivoted_df['SUM'] = pivoted_df.sum(axis=1)
            pivoted_output_filename = 'panel_matrix_solution.csv'
            pivoted_df.to_csv(pivoted_output_filename)
            print(f"📄 Pivoted matrix saved to '{pivoted_output_filename}'")

            # 2. Create the Bill of Quantities (BOQ) Output
            print("\nGenerating Project Bill of Quantities (BOQ)...")
            boq_df = solution_df.groupby('ControllerName')['Quantity'].sum().reset_index()
            # Merge with original controller data to get costs
            boq_df = boq_df.merge(controllers, left_on='ControllerName', right_on='Name', how='left')
            boq_df['TotalCost'] = boq_df['Quantity'] * boq_df['Cost']
            boq_df = boq_df[['ControllerName', 'Quantity', 'Cost', 'TotalCost']] # Reorder columns
            
            # Add a Grand Total row
            grand_total = boq_df['TotalCost'].sum()
            boq_df.loc['Grand Total'] = pd.Series(boq_df['TotalCost'].sum(), index=['TotalCost'])

            boq_output_filename = 'project_boq.csv'
            boq_df.to_csv(boq_output_filename)
            print(f"🧾 Project BOQ saved to '{boq_output_filename}'")

            print(f"\n🎉 All panels processed successfully!")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure 'controllers.csv' and 'panels.csv' are in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")