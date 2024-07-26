() =>{
  const menu = document.querySelector("#menu");
  const menuButton = menu.querySelector("button");

  // scroll on accordion click
  menuButton.addEventListener("click", () => {
    requestAnimationFrame(() => {
      menu.scrollIntoView({ behavior: "instant" });
    });
  });
}
