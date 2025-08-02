const statusDot = document.getElementById("statusDot");

async function healthCheck() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.status === "ok") {
      statusDot.textContent = "🟢";
    } else {
      statusDot.textContent = "🔴";
    }
  } catch {
    statusDot.textContent = "🔴";
  }
}

async function search() {
  const query = document.getElementById("query").value.trim();
  const type = document.getElementById("searchType").value;
  const resultBox = document.getElementById("result");
  if (!query) {
    resultBox.textContent = "⚠️ Please enter a name.";
    return;
  }

  resultBox.textContent = "Loading...";
  try {
    const res = await fetch(`/api/search_${type}?query=${encodeURIComponent(query)}`);
    const data = await res.json();
    const display = data.results.map(r => `${r.category.padEnd(20)} ➜ ${r.max_confidence_score}`).join("\n");
    resultBox.textContent = `Query: ${data.query}\n\n${display}`;
  } catch (e) {
    resultBox.textContent = "❌ Error contacting server.";
  }
}

async function refreshIndexes() {
  const resultBox = document.getElementById("result");
  resultBox.textContent = "🔄 Rebuilding indexes...";
  try {
    const res = await fetch("/api/refresh_indexes", { method: "POST" });
    const data = await res.json();
    resultBox.textContent = `✅ Indexes refreshed: ${data.types.join(", ")}`;
  } catch (e) {
    resultBox.textContent = "❌ Failed to refresh indexes.";
  }
}

healthCheck();
