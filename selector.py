# bms_selector_interactive.py

import pandas as pd
import pulp

# The core optimization function remains unchanged
def find_optimal_combination(panel_requirements, components_df):
    """
    Solves for the most cost-effective combination of components (controllers or modules).
    """
    prob = pulp.LpProblem(f"BMS_Combination_{panel_requirements.name}", pulp.LpMinimize)

    # --- DECISION VARIABLES ---
    component_qty_vars = pulp.LpVariable.dicts("Qty", components_df['Name'], lowBound=0, cat='Integer')
    uio_as_input_vars = pulp.LpVariable.dicts("UIO_as_Input", components_df['Name'], lowBound=0, cat='Continuous')
    uio_as_output_vars = pulp.LpVariable.dicts("UIO_as_Output", components_df['Name'], lowBound=0, cat='Continuous')

    # --- OBJECTIVE FUNCTION (Minimize Cost of selectable components) ---
    prob += pulp.lpSum(
        [components_df.loc[i, 'Cost'] * component_qty_vars[components_df.loc[i, 'Name']] for i in components_df.index]
    ), "Total_Component_Cost"

    # --- CONSTRAINTS ---
    for i in components_df.index:
        model_name = components_df.loc[i, 'Name']
        available_uio = components_df.loc[i, 'UIO'] * component_qty_vars[model_name]
        prob += uio_as_input_vars[model_name] + uio_as_output_vars[model_name] <= available_uio, f"UIO_Allocation_{model_name}"

    total_required_inputs = panel_requirements['DI'] + panel_requirements['AI']
    total_required_outputs = panel_requirements['DO'] + panel_requirements['AO']

    total_provided_inputs = pulp.lpSum(
        [(components_df.loc[i, 'DI'] + components_df.loc[i, 'AI'] + components_df.loc[i, 'UI']) * component_qty_vars[components_df.loc[i, 'Name']] for i in components_df.index]
    ) + pulp.lpSum(uio_as_input_vars)
    
    total_provided_outputs = pulp.lpSum(
        [(components_df.loc[i, 'DO'] + components_df.loc[i, 'AO'] + components_df.loc[i, 'UO']) * component_qty_vars[components_df.loc[i, 'Name']] for i in components_df.index]
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
                component_name = v.name.replace("Qty_", "").replace("_", " ")
                solution[component_name] = int(v.varValue)
        return total_cost, solution
    else:
        return None, None

# --- Main Script Execution ---
if __name__ == "__main__":
    try:
        # Load all three data files
        controllers = pd.read_csv('controllers.csv').fillna(0)
        server_modules = pd.read_csv('server_modules.csv').fillna(0)
        panels = pd.read_csv('panels.csv')
        
        print("✅ Successfully loaded all CSV files.\n")

        # --- INTERACTIVE PROMPT FOR SERVER SELECTION ---
        panel_names = panels['PanelName'].tolist()
        print("Available Panels:")
        print(", ".join(panel_names))
        
        user_input = input("\nEnter the names of the panels to be Server Panels (comma-separated), then press Enter:\n> ")
        
        # Create a set of server panel names for fast lookup
        server_panel_names = {name.strip() for name in user_input.split(',') if name.strip()}
        
        if server_panel_names:
            print(f"\n--> Treating the following as Server Panels: {', '.join(sorted(list(server_panel_names)))}")
        else:
            print("\n--> No server panels selected. Treating all panels as Standard.")

        # --- Automation Server Setup ---
        automation_server_name = 'AS-P-Server'
        asp_info = server_modules[server_modules['Name'] == automation_server_name]
        if asp_info.empty and server_panel_names:
             raise ValueError(f"'{automation_server_name}' not found in server_modules.csv, but server panels were selected!")
        io_modules = server_modules[server_modules['Name'] != automation_server_name].reset_index(drop=True)
        
        panels.set_index('PanelName', inplace=True)
        all_solutions = []

        for panel_name, requirements in panels.iterrows():
            # Check if the current panel was selected by the user to be a server
            is_server_panel = panel_name in server_panel_names
            panel_type = "Server" if is_server_panel else "Standard"

            print("-" * 40)
            print(f"🔍 Solving for Panel: {panel_name} (Type: {panel_type})")

            # --- TWO-TRACK LOGIC ---
            if is_server_panel:
                module_cost, result = find_optimal_combination(requirements, io_modules)
                if result is not None:
                    print(f"✅ Optimal I/O Modules Found!")
                    all_solutions.append({'PanelName': panel_name, 'ControllerName': automation_server_name, 'Quantity': 1})
                    for component, qty in result.items():
                        all_solutions.append({'PanelName': panel_name, 'ControllerName': component, 'Quantity': qty})
                else:
                    all_solutions.append({'PanelName': panel_name, 'ControllerName': 'No Solution Found', 'Quantity': 0})
            else: # Standard Panel
                _, result = find_optimal_combination(requirements, controllers)
                if result:
                    print(f"✅ Optimal Controllers Found!")
                    for component, qty in result.items():
                        all_solutions.append({'PanelName': panel_name, 'ControllerName': component, 'Quantity': qty})
                else:
                    all_solutions.append({'PanelName': panel_name, 'ControllerName': 'No Solution Found', 'Quantity': 0})
        
        if not all_solutions:
            print("No solutions were found for any panels.")
        else:
            # --- GENERATE AND SAVE OUTPUTS (No changes needed here) ---
            solution_df = pd.DataFrame(all_solutions)
            print("\nGenerating Pivoted Panel Matrix...")
            pivoted_df = solution_df.pivot_table(index='PanelName', columns='ControllerName', values='Quantity', fill_value=0)
            pivoted_df['SUM'] = pivoted_df.sum(axis=1)
            pivoted_df.to_csv('panel_matrix_solution.csv')
            print(f"📄 Pivoted matrix saved to 'panel_matrix_solution.csv'")

            print("\nGenerating Project Bill of Quantities (BOQ)...")
            boq_df = solution_df.groupby('ControllerName')['Quantity'].sum().reset_index()
            all_components = pd.concat([controllers, server_modules])
            boq_df = boq_df.merge(all_components, left_on='ControllerName', right_on='Name', how='left')
            boq_df['TotalCost'] = boq_df['Quantity'] * boq_df['Cost']
            boq_df = boq_df[['ControllerName', 'Quantity', 'Cost', 'TotalCost']]
            grand_total = boq_df['TotalCost'].sum()
            boq_df.loc['Grand Total'] = pd.Series(boq_df['TotalCost'].sum(), index=['TotalCost'])
            boq_df.to_csv('project_boq.csv')
            print(f"🧾 Project BOQ saved to 'project_boq.csv'")
            print(f"\n🎉 All panels processed successfully!")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure all three CSV files are in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")