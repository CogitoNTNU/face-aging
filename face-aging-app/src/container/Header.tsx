import React from "react";
import cogito_logo from "../assets/cogito_logo.svg";

export default function Header() {
  return (
    <div>
      <img id="header-logo" src={cogito_logo} alt="logo" />
    </div>
  );
}
