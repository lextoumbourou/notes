// Run this once the DOM has loaded
document.addEventListener("DOMContentLoaded", function () {
  // Ensure all collapsible callouts are collapsed by default
  const collapsibleCallouts = document.querySelectorAll(".callout.is-collapsible");
  collapsibleCallouts.forEach(callout => {
    callout.classList.add("is-collapsed");
    const content = callout.querySelector(".callout-content");
    if (content) content.style.display = "none";
    const foldButton = callout.querySelector(".callout-fold");
    if (foldButton) {
      foldButton.classList.add("is-collapsed");
      foldButton.textContent = "▸"; // Default collapsed icon
    }
  });

  // Attach a click handler to the entire page (event delegation)
  document.addEventListener("click", function (event) {
    // Check if the clicked element or any of its parents is within the .callout-title
    const header = event.target.closest(".callout-title");
    if (!header) return;

    // Find the nearest .callout.is-collapsible container
    const callout = header.closest(".callout.is-collapsible");
    if (!callout) return;

    // Toggle the 'is-collapsed' class on the callout
    callout.classList.toggle("is-collapsed");

    // Grab the callout-content to show/hide
    const content = callout.querySelector(".callout-content");
    if (!content) return;

    // Update visibility and toggle icon
    const foldButton = callout.querySelector(".callout-fold");
    if (callout.classList.contains("is-collapsed")) {
      content.style.display = "none";
      if (foldButton) foldButton.textContent = "▸"; // Update icon to collapsed state
    } else {
      content.style.display = "";
      if (foldButton) foldButton.textContent = "▾"; // Update icon to expanded state
    }
  });
});
