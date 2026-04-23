import os
import json
import argparse
from openai import OpenAI
from agent_core.tools.env_manager import EnvManagerTool
from agent_core.tools.benchmark_runner import BenchmarkRunnerTool
from agent_core.config.sandbox import validate_path

# Replace this with environment variable in production
DEEPSEEK_API_KEY = "sk-e55867361d9b444f88bfe9993c316909"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

class DiagnosticAgent:
    def __init__(self, workspace, venv_path, backend):
        self.workspace = workspace
        self.venv_path = venv_path
        self.backend = backend
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        
        self.env_tool = EnvManagerTool(venv_path)
        self.runner_tool = BenchmarkRunnerTool(workspace)
        
        # Load system prompt
        prompt_path = os.path.join(workspace, "agent_core", "prompts", "system_prompt.md")
        with open(prompt_path, "r") as f:
            self.system_prompt = f.read()

        self.messages = [{"role": "system", "content": self.system_prompt}]
        
    def _call_llm(self):
        """Helper to call DeepSeek Chat API."""
        # Note: We are using deepseek-chat, if deepseek-coder is needed, we can switch
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=self.messages,
            temperature=0.2,
            max_tokens=2048,
            # Tools can be explicitly provided here if we use OpenAI Tool calling format.
            # For simplicity in this script, we'll parse JSON blocks from the response text.
        )
        msg = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": msg})
        return msg

    def _execute_action(self, action_type, payload):
        """Execute a tool action requested by the LLM."""
        print(f"\n[Agent Action Executing]: {action_type} - {payload}")
        try:
            if action_type == "read_file":
                path = payload.get("path")
                validate_path(path)
                with open(path, 'r') as f:
                    return f.read()
                    
            elif action_type == "grep_code":
                keyword = payload.get("keyword")
                path = payload.get("path")
                validate_path(path)
                import subprocess
                result = subprocess.run(["grep", "-rn", keyword, path], capture_output=True, text=True)
                return result.stdout[:2000] # Limit output size
                
            elif action_type == "install_package":
                spec = payload.get("package_spec")
                return json.dumps(self.env_tool.install_package(spec))
                
            elif action_type == "list_packages":
                return json.dumps(self.env_tool.list_packages())
                
            elif action_type == "run_single_test":
                suite = payload.get("suite")
                test = payload.get("test")
                python_bin = os.path.join(self.venv_path, "bin", "python")
                return json.dumps(self.runner_tool.run_single_test(python_bin, self.backend, suite, test))
            
            else:
                return f"Error: Unknown action type {action_type}"
                
        except Exception as e:
            return f"Action execution failed: {str(e)}"

    def run_diagnostics(self, failed_test_info):
        """Main loop for the diagnostic agent."""
        print(f"Starting Diagnostic Agent for backend: {self.backend}")
        
        initial_context = f"""
        A failure has been detected.
        Backend: {self.backend}
        Virtual Environment: {self.venv_path}
        
        Failed Test Info:
        {json.dumps(failed_test_info, indent=2)}
        
        Please begin your TRIAGE and investigation.
        You MUST use the exact action names below when outputting a JSON block:
        
        Available Actions:
        1. "read_file": {{"action": "read_file", "payload": {{"path": "/root/autodl-tmp/..."}}}}
        2. "grep_code": {{"action": "grep_code", "payload": {{"keyword": "searchTerm", "path": "/root/autodl-tmp/..."}}}}
        3. "list_packages": {{"action": "list_packages", "payload": {{}}}}
        4. "install_package": {{"action": "install_package", "payload": {{"package_spec": "mindspore==2.3.0"}}}}
        5. "run_single_test": {{"action": "run_single_test", "payload": {{"suite": "module_smoke", "test": "test_maxpool2d_module"}}}}
        
        Example Output:
        ```json
        {{
            "action": "grep_code",
            "payload": {{"keyword": "MaxPool2D", "path": "/root/autodl-tmp/mindnlp/src/torch4ms"}}
        }}
        ```
        """
        self.messages.append({"role": "user", "content": initial_context})
        
        max_turns = 10
        for turn in range(max_turns):
            print(f"\n--- Agent Turn {turn + 1} ---")
            response = self._call_llm()
            print(response)
            
            if "[STATE: REPORTING]" in response:
                print("\n[Agent Finished] Diagnostic report generated.")
                break
                
            # Check if LLM wants to execute an action
            if "```json" in response:
                try:
                    # Extract JSON block
                    json_str = response.split("```json")[1].split("```")[0].strip()
                    action_req = json.loads(json_str)
                    
                    action_result = self._execute_action(action_req["action"], action_req["payload"])
                    
                    # Feed result back to LLM
                    self.messages.append({
                        "role": "user", 
                        "content": f"Action Result:\n{action_result}\n\nWhat is your next step?"
                    })
                except Exception as e:
                    self.messages.append({
                        "role": "user",
                        "content": f"Failed to parse or execute your action JSON: {e}. Please ensure correct format."
                    })
            else:
                self.messages.append({
                    "role": "user",
                    "content": "Please continue. Remember to output an action JSON if you need to search code or mutate the environment, or transition to REPORTING if you are done."
                })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--venv", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--error", required=True)
    args = parser.parse_args()
    
    test_info = {
        "suite_name": args.suite,
        "test_name": args.test,
        "error_message": args.error
    }
    
    agent = DiagnosticAgent(
        workspace="/root/autodl-tmp/torchbridgebench",
        venv_path=args.venv,
        backend=args.backend
    )
    
    agent.run_diagnostics(test_info)
