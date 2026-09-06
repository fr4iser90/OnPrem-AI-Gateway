/** Public aliases — badge toggles + Preferred filtered to Active model hosts. */
(function () {
  var root = document.querySelector("[data-aliases-root]");
  if (!root) return;

  var list = document.getElementById("alias-list");
  var addForm = document.getElementById("alias-add-form");
  var addStatus = document.getElementById("alias-add-status");
  var meta = {};
  try {
    meta = JSON.parse(root.getAttribute("data-alias-meta") || "{}") || {};
  } catch (e) {
    meta = {};
  }

  function hostsFor(mid) {
    var entry = meta[mid];
    if (!entry || !entry.sources) return [];
    return entry.sources.slice();
  }

  function kindFor(mid) {
    var entry = meta[mid];
    return entry && entry.kind ? entry.kind : "";
  }

  function fillPreferred(select, mid, keep) {
    if (!select) return;
    var hosts = hostsFor(mid);
    var want = keep && hosts.indexOf(keep) >= 0 ? keep : "";
    select.innerHTML = "";
    var empty = document.createElement("option");
    empty.value = "";
    empty.textContent = "—";
    select.appendChild(empty);
    hosts.forEach(function (name) {
      var opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      select.appendChild(opt);
    });
    select.value = want;
  }

  function filterCandPick(card) {
    var pick = card.querySelector("[data-alias-cand-pick]");
    var target = card.querySelector("[data-alias-target]");
    if (!pick || !target) return;
    var kind = kindFor(target.value);
    var have = {};
    candMids(card).forEach(function (m) {
      have[m] = true;
    });
    Array.prototype.forEach.call(pick.options, function (opt) {
      if (!opt.value) return;
      var okKind = !kind || kindFor(opt.value) === kind;
      var hide = !!have[opt.value] || !okKind;
      opt.hidden = hide;
      opt.disabled = hide;
    });
    pick.value = "";
  }

  function setStatus(el, msg, isErr) {
    if (!el) return;
    el.hidden = !msg;
    el.textContent = msg || "";
    el.classList.toggle("flash-err", !!isErr);
  }

  function flagOn(card, name) {
    var btn = card.querySelector('[data-alias-flag="' + name + '"]');
    return !!(btn && btn.getAttribute("aria-pressed") === "true");
  }

  function paintFlag(btn, on, labels) {
    btn.setAttribute("aria-pressed", on ? "true" : "false");
    btn.textContent = on ? labels[0] : labels[1];
    btn.classList.remove("ok", "off", "badge-muted");
    if (btn.getAttribute("data-alias-flag") === "enabled") {
      btn.classList.add(on ? "ok" : "off");
    } else if (!on) {
      btn.classList.add("badge-muted");
    }
  }

  function candMids(card) {
    return Array.prototype.map.call(
      card.querySelectorAll("[data-cand]"),
      function (el) {
        return el.getAttribute("data-cand");
      }
    ).filter(Boolean);
  }

  function escAttr(v) {
    if (window.CSS && CSS.escape) return CSS.escape(v);
    return String(v).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
  }

  function addCandLocal(card, mid) {
    if (!mid) return;
    var target = card.querySelector("[data-alias-target]");
    var activeKind = target ? kindFor(target.value) : "";
    if (activeKind && kindFor(mid) && kindFor(mid) !== activeKind) return;
    var wrap = card.querySelector("[data-alias-cand-list]");
    if (!wrap) return;
    if (card.querySelector('[data-cand="' + escAttr(mid) + '"]')) return;
    var empty = wrap.querySelector("[data-cand-empty]");
    if (empty) empty.remove();

    var chip = document.createElement("span");
    chip.className = "grant-source-chip";
    chip.setAttribute("data-cand", mid);
    chip.innerHTML =
      '<span class="badge mono"></span>' +
      '<button type="button" class="grant-source-revoke" data-alias-cand-remove title="Remove">×</button>';
    chip.querySelector(".badge").textContent = mid;
    chip.querySelector("[data-alias-cand-remove]").setAttribute("data-mid", mid);
    wrap.appendChild(chip);

    if (
      target &&
      !Array.prototype.some.call(target.options, function (o) {
        return o.value === mid;
      })
    ) {
      var opt = document.createElement("option");
      opt.value = mid;
      opt.textContent = mid;
      target.appendChild(opt);
    }
    filterCandPick(card);
  }

  function collectForm(card) {
    var fd = new FormData();
    var name = card.querySelector("[data-alias-name]");
    var target = card.querySelector("[data-alias-target]");
    var source = card.querySelector("[data-alias-source]");
    fd.set("alias_id", name ? name.value.trim() : "");
    fd.set("target_model_id", target ? target.value : "");
    fd.set("preferred_source", source ? source.value : "");
    if (flagOn(card, "enabled")) fd.set("enabled", "1");
    if (flagOn(card, "hide")) fd.set("hide_candidates", "1");
    if (flagOn(card, "show")) fd.set("show_backend", "1");
    candMids(card).forEach(function (m) {
      fd.append("candidates", m);
    });
    return fd;
  }

  function replaceCard(card, html) {
    var wrap = document.createElement("div");
    wrap.innerHTML = String(html).trim();
    var next = wrap.querySelector("[data-alias-card]");
    if (!next) return;
    card.replaceWith(next);
    next.open = true;
    wireCard(next);
  }

  function wireCard(card) {
    var target = card.querySelector("[data-alias-target]");
    var source = card.querySelector("[data-alias-source]");
    var pick = card.querySelector("[data-alias-cand-pick]");
    if (target) {
      target.addEventListener("change", function () {
        fillPreferred(source, target.value, source ? source.value : "");
        filterCandPick(card);
      });
    }
    if (pick) {
      pick.addEventListener("change", function () {
        if (!pick.value) return;
        addCandLocal(card, pick.value);
        // filterCandPick already resets to "Add model…"
      });
    }
    filterCandPick(card);
  }

  if (list) {
    Array.prototype.forEach.call(list.querySelectorAll("[data-alias-card]"), wireCard);
  }

  var addTarget = document.querySelector("[data-alias-add-target]");
  var addSource = document.querySelector("[data-alias-add-source]");
  if (addTarget && addSource) {
    addTarget.addEventListener("change", function () {
      fillPreferred(addSource, addTarget.value, "");
    });
    if (addTarget.value) fillPreferred(addSource, addTarget.value, "");
  }

  if (addForm) {
    addForm.addEventListener("submit", function (ev) {
      ev.preventDefault();
      setStatus(addStatus, "Adding…", false);
      var fd = new FormData(addForm);
      fd.set("hide_candidates", "1");
      fd.set("show_backend", "1");
      fetch("/models/aliases/add", {
        method: "POST",
        credentials: "same-origin",
        headers: { "X-Requested-With": "fetch", Accept: "text/html" },
        body: fd,
      })
        .then(function (res) {
          if (!res.ok) {
            return res.json().then(
              function (j) {
                throw new Error((j && j.error) || "Add failed (" + res.status + ")");
              },
              function () {
                throw new Error("Add failed (" + res.status + ")");
              }
            );
          }
          return res.text();
        })
        .then(function (html) {
          var empty = document.getElementById("alias-empty");
          if (empty) empty.remove();
          var wrap = document.createElement("div");
          wrap.innerHTML = String(html).trim();
          var card = wrap.querySelector("[data-alias-card]");
          if (card && list) {
            list.prepend(card);
            card.open = true;
            wireCard(card);
            if (card.scrollIntoView) {
              card.scrollIntoView({ behavior: "smooth", block: "nearest" });
            }
          }
          addForm.reset();
          if (addSource) fillPreferred(addSource, "", "");
          setStatus(addStatus, "Added.", false);
          setTimeout(function () {
            setStatus(addStatus, "", false);
          }, 2000);
        })
        .catch(function (err) {
          setStatus(addStatus, err.message || "Add failed", true);
        });
    });
  }

  root.addEventListener("click", function (ev) {
    var flagBtn = ev.target.closest("[data-alias-flag]");
    if (flagBtn && root.contains(flagBtn)) {
      ev.preventDefault();
      var kind = flagBtn.getAttribute("data-alias-flag");
      var on = flagBtn.getAttribute("aria-pressed") !== "true";
      if (kind === "enabled") paintFlag(flagBtn, on, ["on", "off"]);
      else if (kind === "hide") paintFlag(flagBtn, on, ["hide list", "list all"]);
      else if (kind === "show") paintFlag(flagBtn, on, ["show target", "name only"]);
      return;
    }

    var detectBtn = ev.target.closest("[data-alias-cand-detect]");
    if (detectBtn) {
      ev.preventDefault();
      var cardD = detectBtn.closest("[data-alias-card]");
      if (!cardD) return;
      var targetD = cardD.querySelector("[data-alias-target]");
      var nameD = cardD.querySelector("[data-alias-name]");
      var statusD = cardD.querySelector("[data-alias-status]");
      var midD = targetD ? targetD.value : "";
      if (!midD) {
        setStatus(statusD, "Pick Active model first.", true);
        return;
      }
      setStatus(statusD, "Detecting…", false);
      var fdD = new FormData();
      fdD.set("model_id", midD);
      if (nameD && nameD.value) fdD.set("alias_id", nameD.value.trim());
      var k = kindFor(midD);
      if (k) fdD.set("kind", k);
      fetch("/models/aliases/detect", {
        method: "POST",
        credentials: "same-origin",
        headers: { "X-Requested-With": "fetch", Accept: "application/json" },
        body: fdD,
      })
        .then(function (res) {
          if (!res.ok) {
            return res.json().then(
              function (j) {
                throw new Error((j && j.error) || "Detect failed");
              },
              function () {
                throw new Error("Detect failed (" + res.status + ")");
              }
            );
          }
          return res.json();
        })
        .then(function (data) {
          var models = (data && data.models) || [];
          var before = candMids(cardD).length;
          models.forEach(function (m) {
            addCandLocal(cardD, m);
          });
          var added = candMids(cardD).length - before;
          if (added > 0) {
            setStatus(
              statusD,
              "+" + added + " detected" + (data.prefix ? " (" + data.prefix + "*)" : "") + " — Save to keep",
              false
            );
          } else {
            setStatus(
              statusD,
              models.length ? "Already in pool." : "No similar models.",
              false
            );
          }
        })
        .catch(function (err) {
          setStatus(statusD, err.message || "Detect failed", true);
        });
      return;
    }

    var rem = ev.target.closest("[data-alias-cand-remove]");
    if (rem) {
      ev.preventDefault();
      var cardR = rem.closest("[data-alias-card]");
      var chip = rem.closest("[data-cand]");
      if (chip) chip.remove();
      if (cardR) {
        var wrap = cardR.querySelector("[data-alias-cand-list]");
        if (wrap && !wrap.querySelector("[data-cand]")) {
          var empty = document.createElement("span");
          empty.className = "muted text-sm";
          empty.setAttribute("data-cand-empty", "");
          empty.textContent = "None yet";
          wrap.appendChild(empty);
        }
        filterCandPick(cardR);
      }
      return;
    }

    var saveBtn = ev.target.closest("[data-alias-save]");
    if (saveBtn) {
      ev.preventDefault();
      var cardS = saveBtn.closest("[data-alias-card]");
      if (!cardS) return;
      var id = cardS.getAttribute("data-alias-id");
      var status = cardS.querySelector("[data-alias-status]");
      setStatus(status, "Saving…", false);
      fetch("/models/aliases/" + id + "/save", {
        method: "POST",
        credentials: "same-origin",
        headers: { "X-Requested-With": "fetch", Accept: "text/html" },
        body: collectForm(cardS),
      })
        .then(function (res) {
          if (!res.ok) {
            return res.json().then(
              function (j) {
                throw new Error((j && j.error) || "Save failed");
              },
              function () {
                throw new Error("Save failed (" + res.status + ")");
              }
            );
          }
          return res.text();
        })
        .then(function (html) {
          replaceCard(cardS, html);
        })
        .catch(function (err) {
          setStatus(status, err.message || "Save failed", true);
        });
      return;
    }

    var delBtn = ev.target.closest("[data-alias-delete]");
    if (delBtn) {
      ev.preventDefault();
      var cardD = delBtn.closest("[data-alias-card]");
      if (!cardD) return;
      var label =
        (cardD.querySelector("[data-alias-label]") || {}).textContent || "alias";
      if (!confirm("Delete alias " + label.trim() + "?")) return;
      var idD = cardD.getAttribute("data-alias-id");
      fetch("/models/aliases/" + idD + "/delete", {
        method: "POST",
        credentials: "same-origin",
        headers: { "X-Requested-With": "fetch" },
      })
        .then(function (res) {
          if (!res.ok && res.status !== 204) throw new Error("Delete failed");
          cardD.remove();
          if (list && !list.querySelector("[data-alias-card]")) {
            var p = document.createElement("p");
            p.className = "muted";
            p.id = "alias-empty";
            p.textContent = "No aliases yet.";
            list.appendChild(p);
          }
        })
        .catch(function () {
          alert("Delete failed");
        });
    }
  });
})();
