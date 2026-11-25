const form = document.querySelector("#question-form");
const input = document.querySelector("#question");
const send = document.querySelector("#send");
const examples = [...document.querySelectorAll(".examples button")];
const messages = document.querySelector("#messages");

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function message(role, label, text) {
  const node = element("article", "", "message " + role);
  const body = element("p", text, "message-body");
  node.append(element("p", label, "message-label"), body);
  messages.append(node);
  return { node, body };
}

function nearChatBottom() {
  const page = document.scrollingElement;
  return page.scrollHeight - page.clientHeight - page.scrollTop < 100;
}

async function typeReply(body, answer) {
  const words = answer.match(/\S+\s*|\s+/gu) ?? [];
  const delay = Math.max(45, Math.min(85, 7000 / Math.max(1, words.length)));
  const text = document.createTextNode("");
  body.replaceChildren(text);
  body.classList.add("typing");
  try {
    for (const [index, word] of words.entries()) {
      if (index) await new Promise((resolve) => setTimeout(resolve, delay));
      const follow = nearChatBottom();
      text.appendData(word);
      if (follow) document.scrollingElement.scrollTop = document.scrollingElement.scrollHeight;
    }
  } finally {
    body.classList.remove("typing");
  }
}

function showDetails(node, data) {
  if (data.sources.length) {
    const sources = element("div", "", "sources");
    const list = element("ul");
    for (const source of data.sources) {
      list.append(element("li", source.title + " · " + source.section));
    }
    sources.append(element("strong", "Sources"), list);
    node.append(sources);
  }

  const details = element("details", "", "inspect");
  details.append(element("summary", "Retrieved passages (" + data.retrieved.length + ")"));
  if (!data.retrieved.length) {
    details.append(element("p", "No passages matched this question.", "inspect-note"));
  }
  for (const { chunk, similarity } of data.retrieved) {
    const passage = element("section", "", "passage");
    passage.append(
      element("h3", chunk.title + " · " + chunk.section),
      element("p", chunk.id + " · Cosine similarity " + similarity.toFixed(3)),
      element("p", chunk.text),
    );
    details.append(passage);
  }

  const model = element("details");
  const label = data.model_requested ? "Model input" : "Prepared model input (preview)";
  model.append(
    element("summary", label),
    element("pre", data.model_input.instructions + "\n\n" + data.model_input.input),
  );
  details.append(model);
  node.append(details);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || send.disabled) return;

  message("user", "You", question);
  const { node, body } = message("assistant", "DeskGuard", "Looking up support articles…");
  node.setAttribute("aria-busy", "true");
  input.value = "";
  send.disabled = true;
  examples.forEach((button) => { button.disabled = true; });
  node.scrollIntoView({ block: "nearest" });

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    if (data.status === "preview") {
      node.querySelector(".message-label").textContent = "DeskGuard · Article preview";
    }
    await typeReply(body, data.answer);
    const follow = nearChatBottom();
    showDetails(node, data);
    if (follow) document.scrollingElement.scrollTop = document.scrollingElement.scrollHeight;
  } catch {
    body.textContent = "DeskGuard could not load a reply. Try sending the question again.";
    input.value = question;
  } finally {
    node.removeAttribute("aria-busy");
    send.disabled = false;
    examples.forEach((button) => { button.disabled = false; });
    input.focus({ preventScroll: true });
  }
});

for (const button of examples) {
  button.addEventListener("click", () => {
    input.value = button.textContent.trim();
    input.focus();
  });
}
