"""Switch the risk-report trigger task XML from S4U (needs admin) to
InteractiveToken (registers unelevated — the ExecAgent pattern)."""
import io

PATH = r"C:\Scripts\trigger_risk_report_task.xml"
content = io.open(PATH, encoding="utf-16").read()
assert "<LogonType>S4U</LogonType>" in content
content = content.replace("<LogonType>S4U</LogonType>",
                          "<LogonType>InteractiveToken</LogonType>")
io.open(PATH, "w", encoding="utf-16").write(content)
print("LogonType -> InteractiveToken")
