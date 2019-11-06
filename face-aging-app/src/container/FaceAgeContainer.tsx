import React from "react";

import Header from "./Header";
import AgeSelector from "../components/AgeSelector";
import GenderSelector from "../components/GenderSelector";
import PictureUploader from "../components/PictureUploader";
import PictureViewer from "../components/PictureViewer";

export default function FaceAgeContainer() {
  return (
    <div>
      <Header />
      <PictureUploader />
      <PictureViewer />
      <AgeSelector />
      <GenderSelector />
    </div>
  );
}
