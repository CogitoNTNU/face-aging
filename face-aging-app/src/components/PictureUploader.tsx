import React, { useState } from "react";

export default function PictureUploader() {
  const [image, setImage] = useState("");

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    event.target.files != null
      ? setImage(URL.createObjectURL(event.target.files[0]))
      : setImage("");
  };

  return (
    <div>
      <input type="file" onChange={handleChange} />
      <img src={image} />
    </div>
  );
}
