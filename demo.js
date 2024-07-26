() =>{
  const header = document.querySelector("#intro");
  const menu = document.querySelector("#menu");
  const menuButton = menu.querySelector("button");
  const menuSpan = menuButton.querySelector("span:first-child");

  const updateMenuText = () => {
    const isOpen = menuButton.classList.contains("open");
    menuSpan.textContent = isOpen ? "Close menu" : "Open menu";

    if (isOpen) {
      menu.scrollIntoView({ behavior: "instant" });
    } else {
      header.scrollIntoView({ behavior: "instant" });
    }
  };

  const observer = new MutationObserver(updateMenuText);
  observer.observe(menuButton, { attributes: true, attributeFilter: ["class"] });
  updateMenuText();
}
