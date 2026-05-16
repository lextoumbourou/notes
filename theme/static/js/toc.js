(function () {
  "use strict";

  function initToc() {
    var tocLinks = document.querySelectorAll(".toc-list a[data-anchor]");
    if (!tocLinks.length) return;

    var headings = [];
    tocLinks.forEach(function (link) {
      var anchor = link.dataset.anchor;
      var el = document.getElementById(anchor);
      if (el) headings.push({ el: el, link: link });
    });
    if (!headings.length) return;

    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          var heading = headings.find(function (h) { return h.el === entry.target; });
          if (!heading) return;
          if (entry.isIntersecting) {
            heading.link.classList.add("in-view");
          } else {
            heading.link.classList.remove("in-view");
          }
        });
      },
      { rootMargin: "0px 0px -70% 0px", threshold: 0 }
    );

    headings.forEach(function (h) { observer.observe(h.el); });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initToc);
  } else {
    initToc();
  }
})();
