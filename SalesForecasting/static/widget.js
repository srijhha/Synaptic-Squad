window.PharmaWidget = (function () {
  let apiBase = "";
  let listening = false;
  const synth = window.speechSynthesis;
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

  function el(tag, cls, text){ const e=document.createElement(tag); if(cls) e.className=cls; if(text) e.textContent=text; return e; }
  function speak(t){ if(!synth) return; synth.speak(new SpeechSynthesisUtterance(t)); }

  function addMsg(box, who, html, tool){
    const row = el("div","msg "+who);
    const bubble = el("div","bubble"); bubble.innerHTML = (html||"").replace(/\n/g,"<br/>");
    row.appendChild(bubble);
    if(tool && tool.html_url){
      const a = el("a","plot-link","Open forecast chart"); a.href = apiBase + tool.html_url; a.target="_blank";
      row.appendChild(el("div")); row.appendChild(a);
    }
    box.appendChild(row); box.scrollTop = box.scrollHeight;
  }

  async function send(input, box, talk){
    const msg = input.value.trim(); if(!msg) return;
    addMsg(box,"user",msg);
    input.value="";

    const r = await fetch(apiBase + "/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:msg})});
    const data = await r.json();
    addMsg(box,"bot",data.reply, data.tool_result);
    if(talk) speak(data.reply);
  }

  function init(opts){
    apiBase = opts.apiBase;
    const bubble = el("div","",null); bubble.id="pharma-bubble"; bubble.textContent="🤖";
    const panel  = el("div","",null); panel.id="pharma-panel";

    const header = el("div","",null); header.id="pharma-header";
    header.appendChild(el("div","",opts.agentName || "Assistant"));
    const close = el("button","iconbtn","✕"); close.onclick=()=>panel.style.display="none"; header.appendChild(close);

    const msgs = el("div","",null); msgs.id="pharma-messages";
    addMsg(msgs,"bot",opts.greeting || "Hello!");

    const footer = el("div","",null); footer.id="pharma-input";
    const input  = el("input"); input.placeholder="Ask about sales, forecasts, dashboard…";
    const sendBtn = el("button","iconbtn","➤"); sendBtn.onclick=()=>send(input,msgs,true);
    const micBtn  = el("button","iconbtn","🎙");
    micBtn.onclick=()=>{
      if(!SR){ alert("SpeechRecognition not supported"); return; }
      if(listening) return;
      const rec = new SR(); rec.lang="en-US"; rec.interimResults=false; rec.maxAlternatives=1;
      listening = true; micBtn.textContent="…";
      rec.onresult = e => { input.value = e.results[0][0].transcript; send(input,msgs,true); };
      rec.onend = ()=>{ listening=false; micBtn.textContent="🎙"; };
      rec.start();
    };

    footer.appendChild(input); footer.appendChild(sendBtn); footer.appendChild(micBtn);
    panel.appendChild(header); panel.appendChild(msgs); panel.appendChild(footer);
    bubble.onclick = ()=>{ panel.style.display = (panel.style.display==="block"?"none":"block"); };

    document.body.appendChild(bubble); document.body.appendChild(panel);
  }

  return { init };
})();