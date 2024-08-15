() =>{
  const menu = document.querySelector("#menu");
  const menuButton = menu.querySelector("button");
  const menuSpan = menuButton.querySelector("span:first-child");
  const content = document.querySelector("#content");

  const updateMenu = () => {
    const isOpen = menuButton.classList.contains("open");
    menuSpan.textContent = isOpen ? "Hide menu" : "Show menu";
    content.style.display = isOpen ? "none" : "flex";
  };

  const observer = new MutationObserver(updateMenu);
  observer.observe(menuButton, { attributes: true, attributeFilter: ["class"] });
  updateMenu();
}
